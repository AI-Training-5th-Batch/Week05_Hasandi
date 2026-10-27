# Week 05 — LLMOps RAG Assistant: Guarded and Evaluated

A locally deployed **Retrieval-Augmented Generation (RAG) Assistant** developed as part of the Virtusa Trainee Engineering Programme — Week 5 assignment.

The system answers factory maintenance questions using a predefined maintenance handbook. It combines semantic retrieval, grounded answer generation, a retrieval-confidence guardrail, performance monitoring, and automated evaluation.

The application runs locally on CPU using open-source models, without requiring API keys.

## 1. Project Objectives

- Develop a RAG pipeline for factory maintenance questions.
- Retrieve relevant information from maintenance handbook documents.
- Generate context-grounded answers using a language model.
- Prevent hallucinations using a retrieval-confidence guardrail.
- Monitor latency, estimated token usage, and hypothetical API costs.
- Evaluate retrieval accuracy, answer quality, and refusal behavior.
- Compare naive and grounded prompting strategies.

## 2. Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core implementation |
| Sentence Transformers | Semantic embeddings |
| all-MiniLM-L6-v2 | Embedding model |
| Google FLAN-T5 Small | Answer generation |
| Hugging Face Transformers | Model loading and inference |
| NumPy | Vector similarity calculations |
| Pandas | Evaluation and observability |
| Matplotlib & Seaborn | Data visualization |
| Jupyter Notebook | Development and experimentation |

## 3. System Architecture

```text
           User Question
                 |
                 v
       Sentence Embedding
                 |
                 v
       Semantic Retrieval
        (Top-k = 3)
                 |
                 v
       Similarity Guardrail
        (Threshold = 0.45)
                 |
          +------+------+
          |             |
      Score < 0.45  Score >= 0.45
          |             |
          v             v
       Refuse       Retrieve Context
                        |
                        v
                 Grounded Prompt
                        |
                        v
                   FLAN-T5
                        |
                        v
                 Generated Answer
                        |
                        v
                Response + Trace
```

## 4. Project Structure

```text
Week5_RAG/
|
|-- corpus/
|   |-- 01_machine_register.txt
|   |-- 02_vibration_guidelines.txt
|   |-- 03_maintenance_schedule.txt
|   |-- 04_defect_handling.txt
|   |-- 05_shift_operations.txt
|   |-- 06_safety_lockout.txt
|   |-- 07_sensor_calibration.txt
|   |-- 08_escalation.txt
|
|-- golden_qa.csv
|-- week5_rag_assistant.ipynb
|-- rag_service.py
|-- README.md
|-- .gitignore
```

## 5. Installation and Setup

### Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/Week05_Hasandi.git
cd Week05_Hasandi
```

Replace `YOUR_USERNAME` with your GitHub username.

### Create a virtual environment

```bash
python -m venv .venv
```

Activate the environment on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### Install dependencies

```bash
pip install sentence-transformers transformers torch numpy pandas matplotlib seaborn jupyter ipykernel
```

**Note:** The embedding and generation models are downloaded from Hugging Face during the first execution. An internet connection is required initially.

## 6. Running the RAG Assistant

### Run from the terminal

```bash
python rag_service.py "How often should lathes be serviced?"
```

Example response:

```json
{
  "answer": "every 250 operating hours",
  "refused": false,
  "sources": [
    "03_maintenance_schedule.txt",
    "07_sensor_calibration.txt"
  ],
  "top_sim": 0.758,
  "latency_ms": 243.1
}
```

### Interactive mode

```bash
python rag_service.py
```

Enter maintenance-related questions and type `quit` to exit.

### Run the Jupyter Notebook

```bash
jupyter notebook
```

Open `week5_rag_assistant.ipynb` and execute the cells in order.

## 7. Retrieval-Augmented Generation Pipeline

**Document Chunking:** Eight maintenance handbook documents are split into smaller paragraphs while retaining their source filenames.

**Embedding:** Text chunks are converted into normalized semantic vectors using `all-MiniLM-L6-v2`.

**Retrieval:** Cosine similarity is used to retrieve the top three most relevant passages.

**Generation:** Retrieved context is passed to FLAN-T5 Small to produce grounded answers.

**Guardrail:** If the highest retrieval similarity is below 0.45, the assistant refuses the question before invoking the generation model.

## 8. Guardrail Implementation

The system uses a retrieval-confidence threshold of **0.45**.

For questions outside the maintenance handbook, the assistant returns:

```json
{
  "answer": "I don't know based on the handbook.",
  "refused": true,
  "sources": [],
  "top_sim": 0.331,
  "latency_ms": 18.5
}
```

This approach helps prevent unsupported responses and avoids unnecessary generation.

## 9. Evaluation Results

The RAG assistant was evaluated against **17 golden question-answer pairs**.

| Evaluation Metric | Result |
|---|---|
| Retrieval Hit Rate | 1.000 (100%) |
| Semantic Answer Quality | 0.975 |
| Refusal Accuracy | 1.000 (100%) |

The results indicate strong retrieval and refusal performance on the provided evaluation dataset.

Semantic answer quality measures embedding similarity to reference answers rather than exact factual accuracy.

## 10. Observability Results

Four test queries were evaluated for response latency, estimated prompt tokens, and hypothetical API costs.

| Metric | Result |
|---|---|
| Average Response Latency | 390.1 ms |
| Total Estimated Prompt Tokens | 367 |
| Estimated Hosted API Cost | $0.000184 |
| Refused Query Latency | 30.3 ms |
| Refused Query Prompt Tokens | 0 |

The refused question was processed without invoking the generation model, demonstrating the efficiency of the guardrail.

The token estimates use a simplified characters-per
