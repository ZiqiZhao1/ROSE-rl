import torch
debug_model = False
if debug_model:
    print(f"[DEBUG] debug_model is enabled")

def _get_candidate_tokens_from_vllm_output(vllm_output, tokenizer, candidate_tokens=None, position=None, top_k=5):
    """
    Helper function to get candidate tokens from vLLM output
    
    Args:
        vllm_output: vLLM output object or list of outputs
        tokenizer: Tokenizer
        candidate_tokens: List of candidate tokens, if None will auto-select top-k
        position: Specified position, if None will use the last position
        top_k: When candidate_tokens is None, select top-k tokens
    
    Returns:
        tuple: (candidate_token_ids, candidate_probs, candidate_texts, position)
    """
    # Handle vLLM output, could be single output object or list of outputs
    if isinstance(vllm_output, list):
        # If it's a list, take the first element
        sample = vllm_output[0]
    else:
        # If it's a single output object, use directly
        sample = vllm_output

    # Get logprobs from vLLM output
    logprobs = sample.logprobs

    if logprobs is None:
        raise ValueError("No logprobs in vLLM output, please ensure logprobs is enabled in sampling params")

    # Determine analysis position
    if position is None:
        position = len(logprobs) - 1  # Use last position

    if position >= len(logprobs):
        raise ValueError(f"Position {position} exceeds logprobs length {len(logprobs)}")

    # Get candidate tokens
    if candidate_tokens is None:
        # Auto-select top-k tokens
        position_logprobs = logprobs[position]
        if position_logprobs is None:
            raise ValueError(f"Position {position} has no logprobs info")

        # Sort by logprob to get top-k
        sorted_logprobs = sorted(position_logprobs.items(), key=lambda x: x[1].logprob, reverse=True)
        top_k_logprobs = sorted_logprobs[:top_k]

        candidate_token_ids = [item[0] for item in top_k_logprobs]
        candidate_probs = [torch.exp(torch.tensor(item[1].logprob)) for item in top_k_logprobs]
    else:
        # Use specified candidate tokens
        candidate_token_ids = []
        candidate_probs = []

        for token in candidate_tokens:
            if isinstance(token, str):
                token_id = tokenizer.encode(token, add_special_tokens=False)[0]
            else:
                token_id = token
            candidate_token_ids.append(token_id)

            # Get probability from logprobs
            position_logprobs = logprobs[position]
            if position_logprobs and token_id in position_logprobs:
                candidate_probs.append(torch.exp(torch.tensor(position_logprobs[token_id].logprob)))
            else:
                candidate_probs.append(0.0)

    # Decode token text
    candidate_texts = [tokenizer.decode([token_id]) for token_id in candidate_token_ids]

    return candidate_token_ids, candidate_probs, candidate_texts, position


def analyze_vllm_output_similarity_with_embeddings(vllm_output, vllm_engine, tokenizer, candidate_tokens=None, position=None, top_k=5):
    """
    Analyze similarity between candidate tokens in vLLM output using real token embeddings
    
    Args:
        vllm_output: vLLM output object or list of outputs
        vllm_engine: vLLM LLM instance
        tokenizer: Tokenizer
        candidate_tokens: List of candidate tokens, if None will auto-select top-k
        position: Specified position, if None will use the last position
        top_k: When candidate_tokens is None, select top-k tokens
    
    Returns:
        dict: Dictionary containing similarity matrix, token info, etc.
    """
    # Get candidate tokens
    candidate_token_ids, candidate_probs, candidate_texts, position = _get_candidate_tokens_from_vllm_output(
        vllm_output, tokenizer, candidate_tokens, position, top_k
    )

    # Get token embeddings
    # Get embeddings for all tokens in vocabulary
    vocab_embeddings = get_vocab_embeddings_from_vllm(vllm_engine, tokenizer)
    candidate_embeddings = vocab_embeddings[candidate_token_ids]  # (n_candidates, hidden_size)

    # Calculate cosine similarity matrix
    normalized_embeddings = torch.nn.functional.normalize(candidate_embeddings, p=2, dim=1)
    similarity_matrix = torch.mm(normalized_embeddings, normalized_embeddings.t())  # (n_candidates, n_candidates)

    return {
        'similarity_matrix': similarity_matrix,
        'candidate_tokens': candidate_texts,
        'candidate_token_ids': candidate_token_ids,
        'candidate_probs': candidate_probs,
        'position': position,
        'n_candidates': len(candidate_token_ids)
    }


def _get_embeddings_from_model(model):
    """
    Helper function to extract embedding layer weights from model
    This function needs to be defined at module level to be pickle-serializable
    
    Args:
        model: vLLM model instance
    
    Returns:
        torch.Tensor: Complete vocabulary embedding weights
    """
    import torch
    from vllm.distributed.parallel_state import (
        get_tensor_model_parallel_world_size,
        get_tensor_model_parallel_rank
    )
    from vllm.distributed.communication_op import tensor_model_parallel_all_gather
    
    # Get model's embedding layer
    if hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):
        embed_layer = model.model.embed_tokens
    elif hasattr(model, 'embed_tokens'):
        embed_layer = model.embed_tokens
    else:
        raise ValueError("Cannot find model's embedding layer")

    # Get current rank's embedding weights
    local_weight = embed_layer.weight.data.clone()
    
    # Check if using VocabParallelEmbedding
    is_vocab_parallel = 'VocabParallelEmbedding' in str(type(embed_layer))
    
    # Get TP info
    tp_size = get_tensor_model_parallel_world_size()
    tp_rank = get_tensor_model_parallel_rank()
    if debug_model:
        print(f"[DEBUG] TP rank {tp_rank}/{tp_size}, is_vocab_parallel={is_vocab_parallel}, local_weight shape={local_weight.shape}")
    
    # If using VocabParallelEmbedding and TP > 1, need to gather from all ranks
    if is_vocab_parallel and tp_size > 1:
        # Use vLLM's all_gather API, concatenate along dim=0 (vocab dimension)
        full_weight = tensor_model_parallel_all_gather(local_weight, dim=0)
        if debug_model:
            print(f"[DEBUG] TP rank {tp_rank}: full_weight shape after all_gather = {full_weight.shape}")
        
        return full_weight.cpu()
    
    # If not using TP or TP=1, return directly
    print(f"[DEBUG] No TP or TP=1, returning local weight")
    return local_weight.cpu()


def get_vocab_embeddings_from_vllm(vllm_engine, tokenizer):
    """
    Get embeddings for all tokens in vocabulary
    
    Args:
        vllm_engine: vLLM LLM instance
        tokenizer: Tokenizer
    
    Returns:
        torch.Tensor: Vocabulary embeddings (vocab_size, hidden_size)
    """
    import torch

    # Use apply_model method
    # _get_embeddings_from_model already handles TP all_gather internally
    results = vllm_engine.apply_model(_get_embeddings_from_model)

    vocab_size = len(tokenizer)
    if debug_model:
        print(f"[DEBUG] vocab_size from tokenizer: {vocab_size}")
        print(f"[DEBUG] apply_model returned {len(results)} result(s)")
    
    # apply_model returns a list, each element corresponds to a model replica
    # In TP case, _get_embeddings_from_model already did all_gather internally
    # So each result should be the complete embedding
    # We just take the first one
    full_embeddings = results[0]
    if debug_model:
        print(f"="*50)
        print(f"Final embedding shape: {full_embeddings.shape}")
        print(f"Final embedding first 5 rows:\n{full_embeddings[:5]}")
        print(f"="*50)
    
    return full_embeddings


def print_similarity_analysis(result, compact=False):
    """Print similarity analysis results"""
    if not compact:
        print(f"Analysis position: {result['position']}")
        print(f"Number of candidate tokens: {result['n_candidates']}")

    print(f"Candidate token list:")
    for i, (token, prob) in enumerate(zip(result['candidate_tokens'], result['candidate_probs'])):
        print(f"  {i+1}. '{token}' (probability: {prob:.4f})")

    if not compact and 'similarity_matrix' in result:
        print(f"\nSimilarity matrix:")
        similarity_matrix = result['similarity_matrix']
        for i in range(similarity_matrix.shape[0]):
            row_similarities = []
            for j in range(similarity_matrix.shape[1]):
                if i != j:
                    sim = similarity_matrix[i, j].item()
                    row_similarities.append(f"{sim:.3f}")
                else:
                    row_similarities.append("1.000")
            print(f"  {result['candidate_tokens'][i]}: [{', '.join(row_similarities)}]")