from src.researchmind.utils.llm_client import ResearchMindLLM
from src.researchmind.agent.router import route
from dataclasses import dataclass
import logging
import mlflow
from tqdm import tqdm

logger = logging.getLogger(__name__)

TEST_QUERIES = [
    # search — 20 queries
    ("What is knowledge distillation?", "search"),
    ("How does batch normalization work?", "search"),
    (
        "What is the mathematical formulation of the cross-entropy loss function?",
        "search",
    ),
    (
        "How does batch normalization stabilize training in deep neural networks?",
        "search",
    ),
    ("What are the key architectural components of the Transformer model?", "search"),
    ("How is the F1 score calculated for imbalanced classification tasks?", "search"),
    (
        "What is the role of the attention mechanism in sequence-to-sequence models?",
        "search",
    ),
    (
        "How does dropout regularization prevent overfitting in neural networks?",
        "search",
    ),
    (
        "What is the definition of the BLEU metric in machine translation evaluation?",
        "search",
    ),
    (
        "How does Adam optimizer adapt learning rates for different parameters?",
        "search",
    ),
    ("What is contrastive learning?", "search"),
    (
        "How is the receptive field calculated in convolutional neural networks?",
        "search",
    ),
    (
        "What is the purpose of layer normalization in recurrent neural networks?",
        "search",
    ),
    (
        "How does the softmax function convert logits into probability distributions?",
        "search",
    ),
    ("What is the concept of vanishing gradients in deep network training?", "search"),
    (
        "How is the Intersection over Union (IoU) metric used in object detection?",
        "search",
    ),
    ("What are the standard hyperparameters for training a ResNet-50 model?", "search"),
    (
        "How does data augmentation improve generalization in computer vision tasks?",
        "search",
    ),
    (
        "What is the theoretical basis for the universal approximation theorem?",
        "search",
    ),
    ("How is the perplexity metric defined for language model evaluation?", "search"),
    ("What is the mechanism behind gradient clipping in RNN training?", "search"),
    ("How does the k-nearest neighbors algorithm determine class labels?", "search"),
    # citation — 20 queries
    ("What papers cite Attention Is All You Need?", "citation"),
    ("What work built on BERT?", "citation"),
    (
        "What papers have cited the original Transformer architecture since its publication?",
        "citation",
    ),
    (
        "Which recent works reference the BERT paper for pre-training strategies?",
        "citation",
    ),
    (
        "Find citations for the ResNet-50 architecture in computer vision literature.",
        "citation",
    ),
    (
        "What studies have built upon the findings of the Attention Is All You Need paper?",
        "citation",
    ),
    (
        "Identify papers that cite the original GPT-3 report for language modeling capabilities.",
        "citation",
    ),
    ("Which works reference the CLIP model for vision-language alignment?", "citation"),
    ("Find citations for the DALL-E 2 paper in generative AI research.", "citation"),
    (
        "What papers have cited the original U-Net architecture for image segmentation?",
        "citation",
    ),
    (
        "Identify references to the YOLOv3 paper in object detection benchmarks.",
        "citation",
    ),
    (
        "Which studies cite the original VAE paper for latent variable modeling?",
        "citation",
    ),
    (
        "Find citations for the AlphaGo Zero paper in reinforcement learning contexts.",
        "citation",
    ),
    (
        "What works reference the Swin Transformer for hierarchical vision processing?",
        "citation",
    ),
    (
        "Identify papers that cite the original BERT paper for natural language understanding tasks.",
        "citation",
    ),
    (
        "Which studies reference the ResNeXt architecture for efficient deep learning?",
        "citation",
    ),
    (
        "Find citations for the original GAN paper in generative modeling literature.",
        "citation",
    ),
    (
        "What papers have cited the ViT (Vision Transformer) model for image classification?",
        "citation",
    ),
    (
        "Identify references to the T5 paper for text-to-text transfer learning.",
        "citation",
    ),
    ("Which works cite the original LSTM paper for sequence modeling?", "citation"),
    ("Find citations for the EfficientNet paper in model scaling laws.", "citation"),
    (
        "What studies reference the original RNN paper for recurrent neural networks?",
        "citation",
    ),
    # compare — 20 queries
    ("How does ViT compare to ResNet?", "compare"),
    ("BERT vs GPT for text classification", "compare"),
    (
        "How does the sample efficiency of PPO compare to SAC in high-dimensional continuous control tasks?",
        "compare",
    ),
    (
        "What are the performance differences between Swin Transformer and ConvNeXt on ImageNet-1k?",
        "compare",
    ),
    (
        "Compare the robustness of adversarial training versus randomized smoothing against L-inf attacks.",
        "compare",
    ),
    (
        "How does LoRA compare to QLoRA in terms of memory footprint and downstream task accuracy for LLMs?",
        "compare",
    ),
    (
        "What are the trade-offs between FSDP and DeepSpeed ZeRO-3 for distributed training of large language models?",
        "compare",
    ),
    (
        "How does the inference latency of Mamba compare to standard Transformers on long-sequence modeling tasks?",
        "compare",
    ),
    (
        "Compare the generalization bounds of margin-based losses versus cross-entropy in binary classification.",
        "compare",
    ),
    (
        "What are the differences in OOD detection performance between Mahalanobis distance and energy-based methods?",
        "compare",
    ),
    (
        "How does DALL-E 2 compare to Stable Diffusion in terms of text-image alignment and visual fidelity?",
        "compare",
    ),
    (
        "Compare the computational complexity of exact attention versus linear attention mechanisms in sequence modeling.",
        "compare",
    ),
    (
        "What are the strengths and weaknesses of GNNs compared to MPNNs for molecular property prediction?",
        "compare",
    ),
    (
        "How does the convergence rate of AdamW compare to SGD with momentum in non-convex optimization landscapes?",
        "compare",
    ),
    (
        "Compare the zero-shot transfer capabilities of CLIP versus ALIGN on fine-grained visual recognition benchmarks.",
        "compare",
    ),
    (
        "What are the differences in calibration error between temperature scaling and Platt scaling for ensemble models?",
        "compare",
    ),
    (
        "How does the data efficiency of self-supervised learning via contrastive loss compare to masked image modeling?",
        "compare",
    ),
    (
        "Compare the interpretability of attention maps versus gradient-based saliency maps in vision transformers.",
        "compare",
    ),
    (
        "What are the performance gaps between sparse MoE architectures and dense models at equivalent parameter counts?",
        "compare",
    ),
    (
        "How does the robustness of diffusion models compare to GANs when subjected to distribution shift?",
        "compare",
    ),
    (
        "Compare the effectiveness of gradient clipping versus gradient normalization in stabilizing training for RNNs.",
        "compare",
    ),
    (
        "What are the differences in few-shot learning performance between prototypical networks and relation networks?",
        "compare",
    ),
    (
        "How does the energy consumption of training a ResNet-50 compare to training a ViT-Base on the same dataset?",
        "compare",
    ),
    # gap_detection — 20 queries
    ("What problems in OOD detection remain unsolved?", "gap_detection"),
    ("What are the open challenges in anomaly detection?", "gap_detection"),
    (
        "What are the primary theoretical gaps in understanding the generalization bounds of overparameterized neural networks?",
        "gap_detection",
    ),
    (
        "Which aspects of out-of-distribution detection remain unresolved in current self-supervised learning frameworks?",
        "gap_detection",
    ),
    (
        "Where do current diffusion models fall short in terms of computational efficiency for real-time generation?",
        "gap_detection",
    ),
    (
        "What open problems exist in the robustness of vision transformers against adversarial perturbations?",
        "gap_detection",
    ),
    (
        "Which theoretical limitations hinder the scalability of reinforcement learning algorithms in continuous control tasks?",
        "gap_detection",
    ),
    (
        "What are the unresolved challenges in aligning multi-modal models with human ethical guidelines?",
        "gap_detection",
    ),
    (
        "Where does the current literature lack empirical evidence regarding the interpretability of attention mechanisms?",
        "gap_detection",
    ),
    (
        "What are the critical gaps in benchmarking fairness metrics for deep learning models in healthcare applications?",
        "gap_detection",
    ),
    (
        "Which aspects of federated learning privacy guarantees remain theoretically unproven under heterogeneous data distributions?",
        "gap_detection",
    ),
    (
        "What are the open research directions in reducing the carbon footprint of training large-scale foundation models?",
        "gap_detection",
    ),
    (
        "Where do current graph neural networks fail to capture long-range dependencies in molecular structures?",
        "gap_detection",
    ),
    (
        "What are the unresolved issues in calibrating uncertainty estimates for ensemble-based deep learning methods?",
        "gap_detection",
    ),
    (
        "Which theoretical gaps exist in understanding the convergence behavior of stochastic gradient descent with adaptive learning rates?",
        "gap_detection",
    ),
    (
        "What are the known limitations of current few-shot learning approaches in zero-shot transfer scenarios?",
        "gap_detection",
    ),
    (
        "Where does the field lack standardized evaluation protocols for generative AI safety?",
        "gap_detection",
    ),
    (
        "What are the open problems in achieving stable training dynamics for generative adversarial networks with high-resolution outputs?",
        "gap_detection",
    ),
    (
        "Which aspects of neural architecture search remain computationally prohibitive for large-scale deployment?",
        "gap_detection",
    ),
    (
        "What are the unresolved challenges in transferring knowledge from supervised to unsupervised domains without label leakage?",
        "gap_detection",
    ),
    (
        "Where do current multi-agent reinforcement learning systems fail to achieve cooperative equilibrium in complex environments?",
        "gap_detection",
    ),
    # recent — 20 queries
    ("What are the latest papers on diffusion models?", "recent"),
    ("Most recent advances in DINO?", "recent"),
    (
        "What are the most recent breakthroughs in diffusion model sampling efficiency?",
        "recent",
    ),
    (
        "Latest state-of-the-art results for vision-language models on VQA benchmarks",
        "recent",
    ),
    (
        "Recent advances in training large language models with reduced compute",
        "recent",
    ),
    ("Newly published works on robust adversarial training for CNNs", "recent"),
    (
        "What are the latest developments in few-shot learning for medical imaging?",
        "recent",
    ),
    (
        "Recent papers addressing catastrophic forgetting in continual learning",
        "recent",
    ),
    (
        "State-of-the-art updates for object detection in autonomous driving scenarios",
        "recent",
    ),
    (
        "Latest research on efficient fine-tuning methods for transformer models",
        "recent",
    ),
    (
        "Recent breakthroughs in generative adversarial networks for high-resolution image synthesis",
        "recent",
    ),
    (
        "Newly released benchmarks for evaluating reasoning capabilities in LLMs",
        "recent",
    ),
    (
        "Recent progress in self-supervised learning for audio signal processing",
        "recent",
    ),
    (
        "Latest methods for improving generalization in reinforcement learning agents",
        "recent",
    ),
    ("Recent studies on the interpretability of deep neural networks", "recent"),
    ("Newly published work on federated learning privacy guarantees", "recent"),
    ("Recent advances in neural architecture search for edge devices", "recent"),
    ("Latest techniques for handling class imbalance in deep learning", "recent"),
    (
        "Recent papers on multimodal representation learning for video understanding",
        "recent",
    ),
    ("State-of-the-art results for natural language inference tasks", "recent"),
    (
        "Recent developments in graph neural networks for molecular property prediction",
        "recent",
    ),
    (
        "Latest research on optimizing large-scale distributed training systems",
        "recent",
    ),
]


@dataclass
class Category:
    name: str
    correct_counter: int = 0
    total_counter: int = 0


def run_eval():
    llm = ResearchMindLLM()
    intents_list = ["search", "citation", "compare", "gap_detection", "recent"]
    stats = {i: Category(name=i) for i in intents_list}

    logger.info("Starting phase4 routing evaluation with %d queries", len(TEST_QUERIES))

    mlflow.set_experiment("Phase 4 Evaluation")
    with mlflow.start_run(run_name=f"phase4_routing_eval"):
        # Log Hyperparameters
        mlflow.log_params(
            {
                "model": "qwen3.5:9b",
                "temp": 0.0,
                "tier": "fast",
                "num_test_queries": len(TEST_QUERIES),
            }
        )
        failed_cases = []
        for i, (query, intent) in enumerate(
            tqdm(TEST_QUERIES, desc="Routing eval", total=len(TEST_QUERIES))
        ):
            state = {
                "query": query,
                "intent": "",
                "retrieved_chunks": [],
                "compared_chunks": None,
                "tool_call_history": [],
                "session_id": "",
                "final_answer": None,
            }

            intent_response = route(state, llm)
            predicted_intent = intent_response["intent"]

            # Update Local Stats
            if intent in stats:
                stats[intent].total_counter += 1
                is_correct = predicted_intent == intent
                if is_correct:
                    stats[intent].correct_counter += 1

                # log the incorrect prediction as an MLflow artifact for error analysis
                if not is_correct:
                    failed_cases.append(
                        {
                            "index": i,
                            "query": query,
                            "ground_truth": intent,
                            "predicted": predicted_intent,
                        }
                    )

                # Log a step-by-step metric (Rolling Accuracy)
                rolling_acc = (
                    stats[intent].correct_counter / stats[intent].total_counter
                )
                mlflow.log_metric(f"{intent}_rolling_accuracy", rolling_acc, step=i)

            if (i + 1) % 20 == 0 or (i + 1) == len(TEST_QUERIES):
                logger.info("Processed %d/%d queries", i + 1, len(TEST_QUERIES))

        # Log Final Summary Metrics
        total_correct = sum(c.correct_counter for c in stats.values())
        overall_accuracy = total_correct / len(TEST_QUERIES)
        mlflow.log_metric("overall_accuracy", overall_accuracy)
        #Log the entire list as a single file after the loop finishes
        mlflow.log_dict(failed_cases, "failures/all_failed_queries.json")

        for cat in stats.values():
            cat_acc = (
                (cat.correct_counter / cat.total_counter)
                if cat.total_counter > 0
                else 0
            )
            mlflow.log_metric(f"final_{cat.name}_acc", cat_acc)

    logger.info("Experiment logged to MLflow. Overall accuracy: %.4f", overall_accuracy)
    return stats


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    stats = run_eval()
    for cat in stats.values():
        logger.info(
            "Category: %s, Accuracy: %.2f (%d/%d)",
            cat.name,
            (cat.correct_counter / cat.total_counter) if cat.total_counter > 0 else 0,
            cat.correct_counter,
            cat.total_counter,
        )
