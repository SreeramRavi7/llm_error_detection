"""Prompts shared by everyone. Change them here only, so all stages match."""

MODELS = {
    "qwen1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
    "llama3b": "meta-llama/Llama-3.2-3B-Instruct",   # gated: accept the license on Hugging Face first
}

SYSTEM = "You are a helpful assistant."

TRIVIA_INSTRUCTION = "Answer the question with a short answer only, no explanation.\nQuestion: {q}\nAnswer:"

GSM8K_INSTRUCTION = (
    "Solve the problem step by step. End with the final answer on its own line "
    "in the form '#### <number>'.\nProblem: {q}"
)

VERBAL_CONF = (
    "Question: {q}\nProposed answer: {a}\n"
    "How confident are you that the proposed answer is correct? "
    "Reply with a single number from 0 to 100."
)


def build_messages(dataset, question):
    tmpl = TRIVIA_INSTRUCTION if dataset == "triviaqa" else GSM8K_INSTRUCTION
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": tmpl.format(q=question)}]


def max_new_tokens(dataset):
    return 32 if dataset == "triviaqa" else 384
