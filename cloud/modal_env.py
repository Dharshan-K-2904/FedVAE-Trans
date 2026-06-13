import modal

# 🔥 ADD THIS BLOCK (VERY IMPORTANT)
image = (
    modal.Image.debian_slim()
    .pip_install(
        "torch",
        "numpy",
        "pandas",
        "scikit-learn"
    )
)

app = modal.App("fedvae-trans")