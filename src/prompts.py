"""Prompts shared by everyone. Change them here only, so all stages match."""

MODELS = {
    "qwen1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
    "llama3b": "meta-llama/Llama-3.2-3B-Instruct",   # gated: accept the license on Hugging Face first
}

SYSTEM = "You are a helpful assistant."

TRIVIA_INSTRUCTION = "Answer the question with a short answer only, no explanation.\nQuestion: {q}\nAnswer:"

# Qwen and Llama are both trained on the \boxed{} format, they mostly ignored '####'
GSM8K_INSTRUCTION = (
    "Solve the problem step by step, briefly. "
    "Put the final answer, a single number, within \\boxed{{}}.\nProblem: {q}"
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
    return 32 if dataset == "triviaqa" else 640   # 384 cut off 12% of answers
