"""
dump_reference_states.py
Used for bisecting: runs the exact prompt through the real HF model (AutoGPTQ CPU), takes the hidden
state at each checkpoint (after the embedding, after layer 0, after the last layer, logits of the last
token) and prints the first few values (same style as the old verify_gptq_dequant.py) to compare
by hand with Nim.

Run:
  python3.10 dump_reference_states.py --model TheBloke/deepseek-coder-6.7B-instruct-GPTQ
"""
import argparse
import torch
from transformers import AutoTokenizer

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt", default='def fib(n):\n    """Return the n-th Fibonacci number."""\n    ')
    args = ap.parse_args()

    print(f"Loading {args.model} (CPU, may take a few minutes)...")
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    from auto_gptq import AutoGPTQForCausalLM
    model = AutoGPTQForCausalLM.from_quantized(
        args.model, device="cpu", use_triton=False, use_safetensors=True,
        trust_remote_code=True, disable_exllama=True, use_exllama=False,
        use_exllamav2=False, use_qigen=False, torch_dtype=torch.float16,
    )
    try:
        model.eval()
    except AttributeError:
        model.model.eval()

    ids = tok.encode(args.prompt, add_special_tokens=True)
    print(f"prompt_ids ({len(ids)}): {ids}")

    input_ids = torch.tensor([ids])
    with torch.no_grad():
        out = model(input_ids, output_hidden_states=True)

    hs = out.hidden_states
    # hs[0] = after the embedding (before block 0)
    # hs[1] = after block 0
    # hs[-1] = after finalNorm (before lm_head) - NOTE: HF usually does NOT apply
    #          finalNorm to hidden_states[-1], it is before the last norm. Check
    #          model.model.norm separately if an exact match is needed.
    def show(name, t):
        flat = t[0, -1, :8].tolist()  # last token, first 8 elements
        print(f"{name} last_token[:8] = {flat}")

    show("after_embedding (hidden_states[0])", hs[0])
    show("after_block_0   (hidden_states[1])", hs[1])
    show("after_last_block(hidden_states[-1])", hs[-1])

    logits = out.logits
    print(f"logits last_token[:8] = {logits[0, -1, :8].tolist()}")
    top5 = torch.topk(logits[0, -1], 5)
    print("top5 next-token ids :", top5.indices.tolist())
    print("top5 next-token vals:", top5.values.tolist())
    print("top5 next-token strs:", [tok.decode([i]) for i in top5.indices.tolist()])

if __name__ == "__main__":
    main()