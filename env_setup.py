import os
import torch
import logging

def setup_environment():
    """Setup CUDA environment and PyTorch configurations"""
    try:
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        logger = logging.getLogger(__name__)
        
        # Set CUDA environment variables
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
        os.environ['TORCH_USE_CUDA_DSA'] = '1'
        
        # Configure PyTorch
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision('high')
        
        # Check CUDA availability and setup
        if torch.cuda.is_available():
            # Set default CUDA device
            torch.cuda.set_device(0)
            
            # Log GPU information
            device_name = torch.cuda.get_device_name()
            memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            logger.info(f"Using GPU: {device_name}")
            logger.info(f"GPU Memory: {memory_gb:.2f} GB")
            
            # Clear GPU cache
            torch.cuda.empty_cache()
            
            # Enable CUDA memory stats
            torch.cuda.memory.set_per_process_memory_fraction(0.9)  # Use up to 90% of GPU memory
            
            return True
        else:
            logger.warning("No CUDA device available. Using CPU.")
            return False
            
    except Exception as e:
        logger.error(f"Error setting up environment: {str(e)}")
        raise

def cleanup_environment():
    """Cleanup CUDA resources"""
    try:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            
    except Exception as e:
        logging.error(f"Error during cleanup: {str(e)}")