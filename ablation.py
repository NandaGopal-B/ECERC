"""
Ablation configuration generator.

Use the returned dictionaries to train separate variants:
- full ECERC
- no evidence gate
- no cause encoder
- no evidence-cause attention
- no feature gate
"""

def get_variants():
    return {
        "ECERC-full": dict(
            use_evidence=True, use_cause=True,
            use_attention=True, use_feature_gate=True
        ),
        "without-evidence-gating": dict(
            use_evidence=False, use_cause=True,
            use_attention=True, use_feature_gate=True
        ),
        "without-cause-encoding": dict(
            use_evidence=True, use_cause=False,
            use_attention=True, use_feature_gate=True
        ),
        "without-evidence-cause-attention": dict(
            use_evidence=True, use_cause=True,
            use_attention=False, use_feature_gate=True
        ),
        "without-feature-gating": dict(
            use_evidence=True, use_cause=True,
            use_attention=True, use_feature_gate=False
        ),
    }

if __name__ == "__main__":
    for name, flags in get_variants().items():
        print(name, flags)
