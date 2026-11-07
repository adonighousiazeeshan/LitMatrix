# eval_dataset.py
# Ground-truth benchmark dataset linking questions to your specific PDF files

GOLD_BENCHMARK = [
    {
        "question": "What model architecture does the paper propose?",
        "target_filename": "attention.pdf",  # Update with your exact filename (e.g., attention.pdf)
    },
    {
        "question": "What transfer learning methodology and training stages does ULMFiT introduce for natural language processing?",
        "target_filename": "ulnfit.pdf",     # Update with your exact filename (e.g., ulnfit.pdf)
    },
]