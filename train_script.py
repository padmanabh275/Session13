import os
import torch
import wandb
import shutil
from config import SmolLM2Config
from model import SmolLM2Lightning
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
from pytorch_lightning.loggers import WandbLogger
from env_setup import setup_environment, cleanup_environment

# Set CUDA environment variables before any other CUDA operations
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'

def setup_training():
    """Setup training environment"""
    try:
        if torch.cuda.is_available():
            # Configure CUDA settings
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            torch.set_float32_matmul_precision('high')
            
            # Set default device
            device = torch.device('cuda:0')
            torch.cuda.set_device(device)
            
            # Print GPU info
            print(f"Using GPU: {torch.cuda.get_device_name()}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            return device
    except Exception as e:
        print(f"CUDA setup error: {str(e)}")
    
    print("Using CPU")
    return torch.device('cpu')

def cleanup_training():
    """Cleanup training resources"""
    try:
        # Move model to CPU before cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Clean up wandb
        try:
            wandb.finish()
        except:
            pass
            
    except Exception as e:
        print(f"Cleanup error: {str(e)}")

# Setup CUDA at module level
device = setup_training()

class GenerationMonitorCallback(Callback):
    def __init__(self, prompt="Explain what machine learning is:", sample_every_n_steps=500):
        super().__init__()
        self.prompt = prompt
        self.sample_every_n_steps = sample_every_n_steps
        
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        try:
            if (trainer.global_step + 1) % self.sample_every_n_steps == 0:
                # Switch to eval mode
                pl_module.eval()
                
                with torch.no_grad():
                    # Tokenize prompt
                    inputs = pl_module.tokenizer(
                        self.prompt, 
                        return_tensors="pt",
                        truncation=True,
                        max_length=pl_module.config.model.max_position_embeddings,
                        padding=True
                    ).to(pl_module.device)
                    
                    try:
                        # Generate text with error handling
                        outputs = pl_module.generate(
                            input_ids=inputs.input_ids,
                            attention_mask=inputs.attention_mask,
                            max_length=100,
                            temperature=0.7,
                            top_p=0.9,
                            top_k=50,
                            do_sample=True,
                            pad_token_id=pl_module.tokenizer.pad_token_id,
                            bos_token_id=pl_module.tokenizer.bos_token_id,
                            eos_token_id=pl_module.tokenizer.eos_token_id
                        )
                        
                        # Decode generated text
                        generated_text = pl_module.tokenizer.decode(outputs[0], skip_special_tokens=True)
                        
                        # Print results
                        print(f"\n=== Generation at step {trainer.global_step + 1} ===")
                        print(f"Prompt: {self.prompt}")
                        print(f"Generated: {generated_text}\n")
                        
                    except RuntimeError as e:
                        print(f"\nError during generation at step {trainer.global_step + 1}: {str(e)}")
                        print(f"Input shape: {inputs.input_ids.shape}")
                        print(f"Input device: {inputs.input_ids.device}")
                
                # Switch back to train mode
                pl_module.train()
                
        except Exception as e:
            print(f"\nCallback error at step {trainer.global_step + 1}: {str(e)}")

def init_wandb(project_name, run_name):
    """Initialize WandB with error handling and cleanup"""
    try:
        # Try to clean up any existing wandb directory
        wandb_dir = os.path.join(os.getcwd(), "wandb")
        if os.path.exists(wandb_dir):
            try:
                shutil.rmtree(wandb_dir)
                print("Cleaned up existing wandb directory")
            except Exception as e:
                print(f"Warning: Could not clean up wandb directory: {str(e)}")
        
        # Create fresh wandb directory with proper permissions
        os.makedirs(wandb_dir, exist_ok=True)
        
        # Initialize WandB logger
        logger = WandbLogger(
            project=project_name,
            name=run_name,
            save_dir=os.getcwd(),
            settings=wandb.Settings(start_method="thread")
        )
        return logger
    
    except Exception as e:
        print(f"Error initializing WandB: {str(e)}")
        print("Continuing without WandB logging...")
        return None

def main():
    device = setup_training()
    
    try:
        # Load configuration
        config = SmolLM2Config("config.yaml")
        
        # Initialize model
        model = SmolLM2Lightning(config)
        
        # Phase 1: Initial Training
        print("\n=== Starting Phase 1 Training ===")
        
        # Initialize wandb logger for phase 1 with error handling
        wandb_logger = init_wandb("smol-lm2", "training_run_phase1")
        
        # Setup checkpoint callback for phase 1
        checkpoint_callback = ModelCheckpoint(
            dirpath=config.training.checkpoint_dir,
            filename="smol-lm2-phase1-{epoch:02d}-{train_loss:.2f}",
            save_top_k=3,
            monitor="train_loss",
            mode="min",
            every_n_train_steps=config.training.save_steps
        )
        
        # Setup generation monitoring callback for phase 1
        generation_callback = GenerationMonitorCallback(
            prompt=config.training.sample_prompt,
            sample_every_n_steps=config.training.sample_frequency
        )
        
        # Initialize trainer for phase 1
        trainer_phase1 = pl.Trainer(
            max_steps=config.training.first_phase_steps,
            accelerator=config.hardware.accelerator,
            devices=config.hardware.devices,
            precision=config.hardware.precision,
            logger=wandb_logger,
            callbacks=[checkpoint_callback, generation_callback],
            gradient_clip_val=config.hardware.gradient_clip,
            accumulate_grad_batches=config.training.gradient_accumulation_steps,
            log_every_n_steps=config.training.logging_steps,
            deterministic=False,
            benchmark=True,
            strategy='auto',  # Let PyTorch Lightning handle device strategy
        )
        
        # Train phase 1 with error handling
        try:
            trainer_phase1.fit(model)
        except Exception as e:
            print(f"Error during phase 1 training: {str(e)}")
            raise
        
        # Save phase 1 checkpoint
        phase1_checkpoint_path = os.path.join(config.training.checkpoint_dir, "smol-lm2-phase1-final.ckpt")
        trainer_phase1.save_checkpoint(phase1_checkpoint_path)
        print(f"Phase 1 completed. Model saved to {phase1_checkpoint_path}")
        
        # Clear GPU memory between phases
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Phase 2: Fine-tuning
        print("\n=== Starting Phase 2 Training ===")
        
        # Load the model from phase 1 checkpoint with error handling
        try:
            model = SmolLM2Lightning.load_from_checkpoint(phase1_checkpoint_path, config=config)
        except Exception as e:
            print(f"Error loading checkpoint for phase 2: {str(e)}")
            raise
        
        # Initialize wandb logger for phase 2 with error handling
        wandb_logger = init_wandb("smol-lm2", "training_run_phase2")
        
        # Setup generation monitoring callback with higher frequency for phase 2
        generation_callback = GenerationMonitorCallback(
            prompt=config.training.sample_prompt,
            sample_every_n_steps=config.training.second_phase_sample_frequency
        )
        
        # Initialize trainer for phase 2
        trainer_phase2 = pl.Trainer(
            max_steps=config.training.second_phase_steps,
            accelerator=config.hardware.accelerator,
            devices=config.hardware.devices,
            precision=config.hardware.precision,
            logger=wandb_logger,
            callbacks=[generation_callback],
            gradient_clip_val=config.hardware.gradient_clip,
            accumulate_grad_batches=config.training.gradient_accumulation_steps,
            log_every_n_steps=config.training.logging_steps,
            deterministic=False,
            benchmark=True,
        )
        
        # Train phase 2 with error handling
        try:
            trainer_phase2.fit(model)
        except Exception as e:
            print(f"Error during phase 2 training: {str(e)}")
            raise
        
        # Save final model
        final_checkpoint_path = os.path.join(config.training.checkpoint_dir, "smol-lm2-final.ckpt")
        trainer_phase2.save_checkpoint(final_checkpoint_path)
        print(f"Phase 2 completed. Final model saved to {final_checkpoint_path}")
        
    except Exception as e:
        print(f"\nTraining failed with error: {str(e)}")
        if torch.cuda.is_available():
            print(f"CUDA memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
            print(f"CUDA memory cached: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
        raise
    finally:
        cleanup_training()

if __name__ == "__main__":
    main() 