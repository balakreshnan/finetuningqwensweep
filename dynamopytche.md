# Dynamo implementation Pytche

## Steps

- information

```
sinfo --summarize
sacctmgr show associations user=$USER format=Account,Partition
squeue -u $USER
```

- create folder

```
mkdir -p ~/.config/enroot
touch ~/.config/enroot/.credentials
```

- Create crendentials

```
cat > ~/.config/enroot/.credentials << 'EOF'
machine gxxxxxx login xxx@xxxx.com password xxxxxxxxx
machine nvcr.io login $oauthtoken password xxxxxxxx
EOF
```


- here in the SLURM container for interactive testing

```
srun --account=general_sa \
     --partition=36x2-a01r \
     --nodes=1 \
     --ntasks-per-node=1 \
     --exclusive \
     --time=2:00:00 \
     --job-name=general_sa-finetune:interactive \
     --container-image=nvcr.io/nvidia/nemo:25.11.nemotron_3_nano \
     --container-mount-home \
     --no-container-remap-root \
     --mpi=pmix \
     --pty bash
```

- Start with clear
- list the pids

```
ps aux | grep dynamo.frontend
ps aux | grep "dynamo.vllm"
```

- Setup the hugging face cache

```
# Check what it is
ls -la ~/.cache/huggingface

# Back it up and recreate as directory
mv ~/.cache/huggingface ~/.cache/huggingface.bak
mkdir -p ~/.cache/huggingface
```

- login into huggingface to download models for serving

```
# Retry login
huggingface-cli login --token $HF_TOKEN
```

- if already in a container then deactivate the python environment

```
deactivate
rm -rf ~/venv_dynamo
python3 -m venv ~/venv_dynamo  # no --system-site-packages
source ~/venv_dynamo/bin/activate
pip install "ai-dynamo[vllm]"
```

- to kill the existing process

```
pkill -f "dynamo.frontend"
pkill -f "dynamo.vllm"
```

- Start the front end and backend

```
# Start frontend
python3 -m dynamo.frontend --router-mode kv --discovery-backend file --http-port 8000 &

# Start vLLM worker
python3 -m dynamo.vllm --model Qwen/Qwen3-0.6B --discovery-backend file &
```

- test the serving

```
curl -sS http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen3-0.6B","messages":[{"role":"user","content":"what is quantum physics?"}],"max_tokens":1500}'
```

