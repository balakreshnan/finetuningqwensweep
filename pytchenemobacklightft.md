

- setup space fist
- to avoid disk space issues

```
rm -rf ~/.cache/huggingface/hub/models--nvidia--NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16

# 2. Point future downloads to lustre
export HF_HOME=/lustre/fsw/general_sa/bbalakreshna/.cache/huggingface
export TRANSFORMERS_CACHE=/lustre/fsw/general_sa/bbalakreshna/.cache/huggingface/hub
mkdir -p $HF_HOME/hub
```

- check space

```
du -sh /lustre/fsw/general_sa/bbalakreshna/.cache/huggingface/hub/
df -h /lustre/fsw/general_sa/bbalakreshna/
du -sh /lustre/fsw/general_sa/bbalakreshna/.cache/huggingface/hub/models--*
du -sh ~/.cache/huggingface/hub/ 2>/dev/null
```

- data set fix

```
cat << 'EOF' > /home/bbalakreshna/.local/lib/python3.12/site-packages/nemo_automodel/components/datasets/llm/chat_collate.py
import torch
from torch.utils.data.dataloader import default_collate

def chat_collate_fn(batch):
    """Strip ___PAD_TOKEN_IDS___ metadata before collating."""
    cleaned = [{k: v for k, v in sample.items() if k != "___PAD_TOKEN_IDS___"} for sample in batch]
    return default_collate(cleaned)
EOF
```

```
sed -i '21a\import torch' /home/bbalakreshna/.local/lib/python3.12/site-packages/nemo_automodel/components/datasets/llm/chat_dataset.py

# Replace "return sample" with tensor conversion + metadata strip
sed -i 's/        return sample/        return {k: torch.tensor(v) for k, v in sample.items() if k != "___PAD_TOKEN_IDS___"}/' /home/bbalakreshna/.local/lib/python3.12/site-packages/nemo_automodel/components/datasets/llm/chat_dataset.py
```

- setup the run yaml

```
cat << 'EOF' > /lustre/fsw/general_sa/bbalakreshna/finetune_nemotron.yaml
recipe: nemo_automodel.recipes.llm.train_ft.TrainFinetuneRecipeForNextTokenPrediction

model:
  _target_: nemo_automodel._transformers.NeMoAutoModelForCausalLM.from_pretrained
  pretrained_model_name_or_path: nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16
  trust_remote_code: true

dataset:
  _target_: nemo_automodel.components.datasets.llm.ChatDataset
  path_or_dataset_id: /lustre/fsw/general_sa/bbalakreshna/train.jsonl
  split: train
  seq_length: 2048
  truncation: longest_first
  padding: max_length

dataloader:
  _target_: torch.utils.data.DataLoader
  batch_size: 1
  num_workers: 0
  pin_memory: true

distributed:
  strategy: fsdp2
  ep_size: 4
  moe: {}
  activation_checkpointing: true

loss_fn:
  _target_: nemo_automodel.components.loss.masked_ce.MaskedCrossEntropy

step_scheduler:
  max_steps: 100
  local_batch_size: 1
  global_batch_size: 4
  ckpt_every_steps: 101

checkpoint:
  checkpoint_dir: /tmp/nemo_ckpt

optimizer:
  _target_: torch.optim.AdamW
  lr: 1.0e-5
  weight_decay: 0.01

seed: 42
EOF
```

- create directory

```
mkdir -p /tmp/nemo_ckpt
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p /tmp/nemo_ckpt
```

```
pkill -9 -u bbalakreshna python
nvidia-smi
```

- run the expermiment

```
python -m nemo_automodel.cli.app /lustre/fsw/general_sa/bbalakreshna/finetune_nemotron.yaml
```

```
rm -rf /tmp/nemo_ckpt/*
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -m nemo_automodel.cli.app /lustre/fsw/general_sa/bbalakreshna/finetune_nemotron.yaml
```

- create the data

```
python -c "
import json

samples = [
    ('What is NVIDIA?', 'NVIDIA is a technology company known for designing graphics processing units (GPUs), system-on-chip units, and AI computing platforms.'),
    ('What is CUDA?', 'CUDA is a parallel computing platform and programming model created by NVIDIA that allows developers to use GPUs for general-purpose computing tasks.'),
    ('What is a GPU?', 'A GPU (Graphics Processing Unit) is a specialized processor designed to accelerate graphics rendering and parallel computations.'),
    ('What is TensorRT?', 'TensorRT is NVIDIA SDK for high-performance deep learning inference, optimizing trained models for faster deployment on NVIDIA hardware.'),
    ('What is NVLink?', 'NVLink is a high-bandwidth, energy-efficient interconnect technology by NVIDIA that enables fast communication between GPUs and CPUs.'),
    ('What is NVIDIA DGX?', 'NVIDIA DGX is a line of purpose-built AI supercomputers designed for enterprise AI development, training, and deploying large-scale models.'),
    ('What is Triton Inference Server?', 'Triton Inference Server is an open-source inference serving software by NVIDIA that helps deploy AI models at scale in production environments.'),
    ('What is NVIDIA Omniverse?', 'NVIDIA Omniverse is a platform for building and operating metaverse applications, enabling real-time 3D collaboration and physically accurate simulation.'),
    ('What is deep learning?', 'Deep learning is a subset of machine learning that uses neural networks with many layers to learn complex patterns from large amounts of data.'),
    ('What is a neural network?', 'A neural network is a computing system inspired by biological neural networks, consisting of interconnected nodes that process information in layers.'),
    ('What is transfer learning?', 'Transfer learning is a machine learning technique where a model trained on one task is reused as the starting point for a model on a different but related task.'),
    ('What is fine-tuning?', 'Fine-tuning is the process of taking a pre-trained model and further training it on a smaller, task-specific dataset to adapt it for a particular use case.'),
    ('What is a transformer model?', 'A transformer is a deep learning architecture that uses self-attention mechanisms to process sequential data, forming the basis of models like GPT and BERT.'),
    ('What is NLP?', 'Natural language processing (NLP) is a field of AI that focuses on enabling computers to understand, interpret, and generate human language.'),
    ('What is computer vision?', 'Computer vision is a field of AI that enables computers to interpret and understand visual information from images and videos.'),
    ('What is reinforcement learning?', 'Reinforcement learning is a type of machine learning where an agent learns to make decisions by taking actions in an environment and receiving rewards or penalties.'),
    ('What is supervised learning?', 'Supervised learning is a machine learning approach where the model is trained on labeled data, learning to map inputs to known outputs.'),
    ('What is unsupervised learning?', 'Unsupervised learning is a machine learning approach where the model finds patterns and structure in data without labeled examples.'),
    ('What is a CNN?', 'A convolutional neural network (CNN) is a type of deep learning model specifically designed for processing structured grid data like images.'),
    ('What is an RNN?', 'A recurrent neural network (RNN) is a type of neural network designed for sequential data, where connections between nodes form directed cycles.'),
    ('What is batch normalization?', 'Batch normalization is a technique that normalizes the inputs to each layer in a neural network, improving training stability and speed.'),
    ('What is dropout?', 'Dropout is a regularization technique where randomly selected neurons are ignored during training to prevent overfitting.'),
    ('What is backpropagation?', 'Backpropagation is an algorithm used to train neural networks by computing gradients of the loss function with respect to each weight.'),
    ('What is gradient descent?', 'Gradient descent is an optimization algorithm that iteratively adjusts model parameters in the direction that minimizes the loss function.'),
    ('What is a learning rate?', 'The learning rate is a hyperparameter that controls how much model weights are updated during each step of training.'),
    ('What is an epoch?', 'An epoch is one complete pass through the entire training dataset during the training process of a machine learning model.'),
    ('What is overfitting?', 'Overfitting occurs when a model learns the training data too well, including noise and outliers, resulting in poor performance on unseen data.'),
    ('What is underfitting?', 'Underfitting occurs when a model is too simple to capture the underlying patterns in the data, resulting in poor performance on both training and test data.'),
    ('What is data augmentation?', 'Data augmentation is a technique of artificially increasing the training dataset size by applying transformations like rotation, flipping, or scaling to existing data.'),
    ('What is a loss function?', 'A loss function measures how well a model predictions match the actual target values, guiding the optimization process during training.'),
    ('What is mixed precision training?', 'Mixed precision training uses both 16-bit and 32-bit floating point types during model training to reduce memory usage and improve performance.'),
    ('What is model parallelism?', 'Model parallelism is a technique where different parts of a neural network are distributed across multiple GPUs or devices for training.'),
    ('What is data parallelism?', 'Data parallelism is a technique where the same model is replicated across multiple GPUs, each processing different batches of data simultaneously.'),
    ('What is pipeline parallelism?', 'Pipeline parallelism divides a model into stages, with each stage running on a different GPU, and data flowing through the pipeline in micro-batches.'),
    ('What is expert parallelism?', 'Expert parallelism distributes different experts of a Mixture of Experts model across multiple GPUs, allowing each GPU to handle a subset of experts.'),
    ('What is MoE?', 'A Mixture of Experts (MoE) model is an architecture where different subnetworks (experts) specialize in different parts of the input space, with a gating mechanism selecting which experts to use.'),
    ('What is FSDP?', 'Fully Sharded Data Parallel (FSDP) is a training technique that shards model parameters, gradients, and optimizer states across data-parallel workers to reduce memory usage.'),
    ('What is FSDP2?', 'FSDP2 is the next generation of Fully Sharded Data Parallel in PyTorch, offering improved composability with tensor parallelism and other parallelism strategies.'),
    ('What is tensor parallelism?', 'Tensor parallelism splits individual tensors across multiple GPUs, enabling training of models too large to fit on a single device.'),
    ('What is sequence parallelism?', 'Sequence parallelism distributes the processing of long sequences across multiple GPUs, reducing memory requirements for attention computations.'),
    ('What is flash attention?', 'Flash attention is an efficient attention algorithm that reduces memory usage from quadratic to linear in sequence length while being faster than standard attention.'),
    ('What is the Adam optimizer?', 'Adam is an adaptive learning rate optimization algorithm that combines the advantages of AdaGrad and RMSProp for efficient deep learning training.'),
    ('What is weight decay?', 'Weight decay is a regularization technique that adds a penalty proportional to the magnitude of model weights, preventing them from growing too large.'),
    ('What is a tokenizer?', 'A tokenizer converts raw text into numerical tokens that a language model can process, splitting text into words, subwords, or characters.'),
    ('What is BPE?', 'Byte Pair Encoding (BPE) is a tokenization algorithm that iteratively merges the most frequent pairs of characters or subwords to build a vocabulary.'),
    ('What is attention?', 'The attention mechanism allows a model to focus on relevant parts of the input when producing each part of the output, weighing the importance of different elements.'),
    ('What is self-attention?', 'Self-attention is a mechanism where each element in a sequence attends to all other elements, computing relationships and dependencies within the same sequence.'),
    ('What is multi-head attention?', 'Multi-head attention runs multiple attention operations in parallel, allowing the model to attend to information from different representation subspaces.'),
    ('What is an LLM?', 'A large language model (LLM) is a neural network trained on massive text datasets that can understand and generate human-like text across many tasks.'),
    ('What is GPT?', 'GPT (Generative Pre-trained Transformer) is a family of autoregressive language models that generate text by predicting the next token in a sequence.'),
    ('What is BERT?', 'BERT (Bidirectional Encoder Representations from Transformers) is a language model that uses bidirectional context to understand the meaning of words in text.'),
    ('What is inference?', 'Inference is the process of using a trained model to make predictions or generate outputs on new, previously unseen data.'),
    ('What is quantization?', 'Model quantization reduces the precision of model weights and activations from 32-bit to lower bit-widths like 8-bit or 4-bit to reduce memory and improve speed.'),
    ('What is FP8?', 'FP8 training uses 8-bit floating point precision for computations during model training, offering significant speedups with minimal accuracy loss on modern hardware.'),
    ('What is BF16?', 'BF16 (Brain Floating Point 16) is a 16-bit floating point format that maintains the same exponent range as FP32 while using fewer mantissa bits, ideal for deep learning.'),
    ('What is activation checkpointing?', 'Activation checkpointing is a memory optimization technique that recomputes intermediate activations during the backward pass instead of storing them all in memory.'),
    ('What is gradient accumulation?', 'Gradient accumulation is a technique where gradients from multiple mini-batches are accumulated before updating model weights, effectively simulating larger batch sizes.'),
    ('What is distributed training?', 'Distributed training splits the training workload across multiple GPUs or machines to train larger models faster than a single device could.'),
    ('What is NCCL?', 'NCCL (NVIDIA Collective Communication Library) is a library of multi-GPU and multi-node communication primitives optimized for NVIDIA GPUs.'),
    ('What is NeMo?', 'NeMo is NVIDIA open-source framework for building, training, and fine-tuning large-scale AI models including language, speech, and vision models.'),
    ('What is DeepSpeed?', 'DeepSpeed is a deep learning optimization library by Microsoft that enables efficient distributed training with memory optimization techniques like ZeRO.'),
    ('What is PyTorch?', 'PyTorch is an open-source deep learning framework that provides tensor computation with GPU acceleration and automatic differentiation for building neural networks.'),
    ('What is TensorFlow?', 'TensorFlow is an open-source machine learning framework developed by Google that provides tools for building, training, and deploying machine learning models.'),
    ('What is Hugging Face?', 'Hugging Face is a platform and community that provides open-source libraries, pre-trained models, and datasets for natural language processing and machine learning.'),
    ('What is a pre-trained model?', 'A pre-trained model is a neural network that has been previously trained on a large dataset and can be adapted for specific tasks through fine-tuning.'),
    ('What is LoRA?', 'LoRA (Low-Rank Adaptation) is a parameter-efficient fine-tuning technique that adds small trainable rank decomposition matrices to frozen pre-trained model weights.'),
    ('What is PEFT?', 'PEFT (Parameter-Efficient Fine-Tuning) refers to methods that fine-tune only a small subset of model parameters while keeping most of the pre-trained weights frozen.'),
    ('What is RLHF?', 'RLHF (Reinforcement Learning from Human Feedback) is a training technique that uses human preferences to align language models with desired behaviors.'),
    ('What is prompt engineering?', 'Prompt engineering is the practice of designing and optimizing input prompts to effectively guide AI models toward producing desired outputs.'),
    ('What is RAG?', 'RAG (Retrieval-Augmented Generation) is a technique that combines a retrieval system with a generative model, allowing it to access external knowledge during generation.'),
    ('What is a vector database?', 'A vector database is a specialized database that stores and indexes high-dimensional vectors, enabling efficient similarity search for AI applications.'),
    ('What is an embedding?', 'An embedding is a dense vector representation of data in a continuous vector space where similar items are closer together.'),
    ('What is cosine similarity?', 'Cosine similarity measures the cosine of the angle between two vectors, commonly used to determine how similar two embeddings or documents are.'),
    ('What is a hyperparameter?', 'A hyperparameter is a configuration value set before training begins that controls the training process, such as learning rate, batch size, and number of layers.'),
    ('What is cross-entropy loss?', 'Cross-entropy loss measures the difference between predicted probability distributions and actual distributions, commonly used in classification tasks.'),
    ('What is softmax?', 'Softmax is a function that converts a vector of raw scores into a probability distribution, where all values sum to one.'),
    ('What is ReLU?', 'ReLU (Rectified Linear Unit) is an activation function that outputs the input directly if positive, and zero otherwise, widely used in neural networks.'),
    ('What is GELU?', 'GELU (Gaussian Error Linear Unit) is an activation function that smoothly approximates ReLU and is commonly used in transformer models.'),
    ('What is layer normalization?', 'Layer normalization normalizes the activations across features for each individual sample, stabilizing and accelerating neural network training.'),
    ('What is a residual connection?', 'A residual connection adds the input of a layer directly to its output, helping gradients flow through deep networks during training.'),
    ('What is positional encoding?', 'Positional encoding adds information about the position of tokens in a sequence to their embeddings, since transformers have no inherent notion of order.'),
    ('What is beam search?', 'Beam search is a decoding algorithm that explores multiple candidate sequences simultaneously to find higher-quality outputs from language models.'),
    ('What is temperature in generation?', 'Temperature is a parameter that controls the randomness of text generation; lower values make output more deterministic, higher values more creative.'),
    ('What is top-k sampling?', 'Top-k sampling restricts text generation to the k most probable next tokens, reducing the chance of generating unlikely or nonsensical words.'),
    ('What is top-p sampling?', 'Top-p (nucleus) sampling selects from the smallest set of tokens whose cumulative probability exceeds p, balancing diversity and quality in generation.'),
    ('What is a GPU cluster?', 'A GPU cluster is a group of interconnected servers equipped with multiple GPUs, designed for large-scale parallel computing and AI model training.'),
    ('What is Kubernetes?', 'Kubernetes is an open-source container orchestration platform that automates deploying, scaling, and managing containerized applications.'),
    ('What is Docker?', 'Docker is a platform for building, shipping, and running applications in containers, providing consistent environments across development and production.'),
    ('What is MLOps?', 'MLOps is a set of practices that combines machine learning, DevOps, and data engineering to deploy and maintain ML models in production reliably.'),
    ('What is a checkpoint?', 'A model checkpoint is a saved snapshot of a model weights and training state at a particular point during training, allowing training to be resumed.'),
    ('What is NVIDIA A100?', 'The NVIDIA A100 is a data center GPU based on the Ampere architecture, designed for AI training and inference with support for multi-instance GPU partitioning.'),
    ('What is NVIDIA H100?', 'The NVIDIA H100 is a data center GPU based on the Hopper architecture, featuring the Transformer Engine for accelerated AI training and inference.'),
    ('What is NVIDIA B200?', 'The NVIDIA B200 is a next-generation GPU based on the Blackwell architecture, delivering massive performance improvements for AI training and inference workloads.'),
    ('What is the Transformer Engine?', 'The Transformer Engine is a component in NVIDIA Hopper GPUs that automatically manages mixed FP8 and FP16 precision for transformer model acceleration.'),
    ('What is cuDNN?', 'cuDNN (CUDA Deep Neural Network Library) is a GPU-accelerated library of primitives for deep neural networks, optimizing standard routines like convolution and pooling.'),
    ('What is NVIDIA RAPIDS?', 'NVIDIA RAPIDS is a suite of open-source GPU-accelerated libraries for data science and analytics, providing pandas-like and scikit-learn-like APIs on GPUs.'),
    ('What is edge AI?', 'Edge AI refers to running AI algorithms locally on hardware devices at the edge of the network, rather than in a centralized cloud, enabling real-time processing.'),
    ('What is federated learning?', 'Federated learning is a machine learning approach where models are trained across multiple decentralized devices while keeping data local for privacy.'),
    ('What is model distillation?', 'Model distillation is a technique where a smaller student model is trained to replicate the behavior of a larger teacher model, reducing size while preserving performance.'),
    ('What is pruning?', 'Pruning removes unnecessary weights or neurons from a neural network to reduce its size and computational cost while maintaining accuracy.'),
    ('What is NVIDIA Jetson?', 'NVIDIA Jetson is a series of embedded computing modules designed for AI at the edge, enabling deployment of neural networks on small, power-efficient devices.'),
    ('What is NVIDIA DRIVE?', 'NVIDIA DRIVE is an end-to-end platform for developing autonomous vehicles, providing hardware, software, and simulation tools for self-driving technology.'),
    ('What is NVIDIA Clara?', 'NVIDIA Clara is a healthcare application framework that includes tools for medical imaging, genomics, and drug discovery powered by GPU-accelerated computing.'),
]

with open('/lustre/fsw/general_sa/bbalakreshna/train.jsonl', 'w') as f:
    for q, a in samples:
        record = {'messages': [{'role': 'user', 'content': q}, {'role': 'assistant', 'content': a}]}
        f.write(json.dumps(record) + '\n')

print(f'Written {len(samples)} samples')
"
```