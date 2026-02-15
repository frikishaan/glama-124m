import torch

def check_flex_attention_support():
    # 1. Check PyTorch Version (Needs 2.5.0+)
    version = torch.__version__
    print(f"PyTorch Version: {version}")
    
    # Simple version check (parsing major.minor)
    major, minor = map(int, version.split('.')[:2])
    if major < 2 or (major == 2 and minor < 5):
        print("❌ Flex Attention requires PyTorch 2.5.0 or later.")
        return False

    # 2. Check CUDA Availability
    if not torch.cuda.is_available():
        print("❌ CUDA is not available. Flex Attention requires a GPU.")
        return False

    # 3. Check GPU Compute Capability (Needs 7.0+ for Volta, usually recommended 8.0+ for best perf)
    # FlexAttention officially supports:
    # - NVIDIA V100 (Volta, sm_70)
    # - NVIDIA A100 (Ampere, sm_80)
    # - NVIDIA H100 (Hopper, sm_90)
    # - Consumer RTX 30/40 series (sm_86, sm_89)
    capability = torch.cuda.get_device_capability()
    gpu_name = torch.cuda.get_device_name()
    print(f"GPU: {gpu_name} (Compute Capability: {capability[0]}.{capability[1]})")
    
    if capability[0] < 7:
        print("❌ GPU Compute Capability is too low. Requires 7.0+ (Volta).")
        return False

    # 4. Try Import
    try:
        from torch.nn.attention.flex_attention import flex_attention
        print("✅ torch.nn.attention.flex_attention is importable.")
        torch.backends.cuda.enable_flash_sdp(enabled=True)
        print("FlashAttention available:", torch.backends.cuda.flash_sdp_enabled())
        print("bfloat16 supported:", torch.cuda.is_bf16_supported())
        return True
    except ImportError:
        print("❌ logic passed but import failed. You might be on an incomplete nightly build.")
        return False

if __name__ == "__main__":
    check_flex_attention_support()
