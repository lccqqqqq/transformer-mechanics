import torch as t
from nnsight import LanguageModel
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForMaskedLM
import os
import plotly.graph_objects as go
from jaxtyping import Float
import inspect
import re
from collections import defaultdict
import sys
import psutil

from typing import Callable
from jaxtyping import Float
from torch import Tensor

# ----- Loading models -----

MODEL_LIST = {
    "llama-8b-r1": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "llama-8b": "meta-llama/Meta-Llama-3-8B",
    "qwen-2.5-1.5b": "Qwen/Qwen2.5-1.5B",
    "gpt2-small": "openai-community/gpt2",
    "gpt2-medium": "openai-community/gpt2-medium",
    "gpt2-large": "openai-community/gpt2-large",
    "gpt2-xl": "openai-community/gpt2-xl",
    "albert": "albert/albert-xlarge-v2"
}


def load_model_and_tokenizer(model_name: str, device: str = "cuda", torch_dtype: t.dtype = t.bfloat16, format = 'nns'):
    """
    Load a model from the model list
    Args:
        model_name: str
        device: str
        torch_dtype: t.dtype
    Returns:
        model: LanguageModel
        tokenizer: AutoTokenizer
    """
    if model_name not in MODEL_LIST:
        raise ValueError(f"Model {model_name} not found in MODEL_LIST")
    
    # load model
    model_id = MODEL_LIST[model_name]
    if format == 'nns':
        try:
            model = LanguageModel(model_id, device_map=device, torch_dtype=torch_dtype, dispatch=True)
            print(f"Model {model_name} loaded using nnsight")
        except Exception as e:
            print(f"Error loading model {model_name} using nnsight: {e}")
            print("Switching to default hf format")
            format = 'hf'
    
    if format == 'hf':
        # using the default hf format
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch_dtype,
                device_map=device,
                trust_remote_code=True,
            )
        except Exception as e:
            print(f"Error loading model {model_name} using hf format: {e}")
            print("Switching to MaskedLM")
            model = AutoModelForMaskedLM.from_pretrained(
                model_id,
                torch_dtype=torch_dtype,
                device_map=device,
                trust_remote_code=True,
            )
    
    # load tokenizer
    if format == 'nns':
        tokenizer = model.tokenizer
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # set to evaluation mode
    model.eval()
    
    return model, tokenizer


def print_model_info(model):
    """
    Print the model information
    """
    print(model.config)

# ----- Saving plots -----

def save_plotly_fig(fig, path, name):
    """
    Save a plotly figure to a specified path with a given name.
    
    Args:
        fig: plotly figure object
        path: directory path to save the figure
        name: filename (without extension)
    """
    os.makedirs(path, exist_ok=True)
    fig.write_html(f"{path}/{name}.html")
    fig.write_image(f"{path}/{name}.png", width=1200, height=800)
    print(f"Figure saved to {path}/{name}.html and {path}/{name}.png")

def plot_hist_and_heatmap(
    similarity_matrix: Float[t.Tensor, "seq_len seq_len"],
    path: str = "plots",
    name: str = "similarity_matrix"
):
    """
    Plot a histogram and a heatmap of the similarity matrix, excluding the diagonal
    """
    # exclude the diagonal
    similarity_matrix = similarity_matrix.clone()

    from plotly.subplots import make_subplots
    
    # Create subplots side by side
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Similarity Matrix Heatmap", "Similarity Distribution"),
        specs=[[{"type": "heatmap"}, {"type": "histogram"}]]
    )
    
    # Add heatmap to first subplot
    fig.add_trace(
        go.Heatmap(z=similarity_matrix.cpu().numpy(), colorscale="Viridis"),
        row=1, col=1
    )
    
    # Add histogram to second subplot
    fig.add_trace(
        go.Histogram(x=similarity_matrix.flatten().cpu().numpy(), nbinsx=50),
        row=1, col=2
    )
    
    # Update layout
    fig.update_layout(
        title="Similarity Matrix Analysis",
        showlegend=False
    )
    
    # Update subplot titles
    fig.update_xaxes(title_text="Column", row=1, col=1)
    fig.update_yaxes(title_text="Row", row=1, col=1)
    fig.update_xaxes(title_text="Similarity Value", row=1, col=2)
    fig.update_yaxes(title_text="Count", row=1, col=2)
    
    fig.show()
    save_plotly_fig(fig, path, name)
    
# ----- general utilities -----

def print_shape(tensor: t.Tensor):
    """
    Simple helper function to print the shape of a tensor.
    
    Args:
        tensor: A PyTorch tensor or any object with a .shape attribute
    
    Example:
        attnout = t.randn(32, 768)
        print_shape(attnout)  # Output: shape of attnout is torch.Size([32, 768])
    """
    # Get the name of the variable from the caller's frame
    frame = inspect.currentframe().f_back
    calling_line = inspect.getframeinfo(frame).code_context[0].strip()
    # Extract variable name from the function call
    # This looks for print_shape(variable_name) pattern
    match = re.search(r'print_shape\((.*?)\)', calling_line)
    if match:
        var_name = match.group(1).strip()
    else:
        var_name = "tensor"
        
    if hasattr(tensor, 'shape'):
        print(f"Shape of [{var_name}]: {tensor.shape}")
    else:
        print(f"{var_name} has no shape attribute. Type: {type(tensor)}")
        
        
# ----- memory monitoring utilities -----

def print_gpu_memory(start_str: str = ""):
    if t.cuda.is_available():
        print(start_str)
        for i in range(t.cuda.device_count()):
            total = t.cuda.get_device_properties(i).total_memory / 1024**3  # Convert to GB
            reserved = t.cuda.memory_reserved(i) / 1024**3
            allocated = t.cuda.memory_allocated(i) / 1024**3
            print(
                f"GPU {i}:",
                f"reserved/allocated/total: {reserved:.2f}/{allocated:.2f}/{total:.2f}",
            )
    else:
        print("No GPU available")

# Example usage:
# memory_info = calculate_tensor_memory((2000, 4096, 200), dtype=t.bfloat16)
# print(f"Memory required: {memory_info['gb']:.2f} GB")

class MemoryMonitor:
    """
    A class to monitor GPU memory usage over time.
    """
    def __init__(self, name="Memory Monitor"):
        """
        Initialize the memory monitor.
        
        Args:
            name: Name of the monitor (default: "Memory Monitor")
        """
        self.name = name
        self.measurements = []  # Will store (time, memory, label) tuples
        self.start_time = None
        self.start_memory = None
        
    def start(self):
        """Start monitoring memory usage."""
        import time
        self.start_time = time.time()
        self.start_memory = t.cuda.memory_allocated() / (1024 * 1024 * 1024)  # GB
        self.measurements.append((0, self.start_memory, "Start"))
        print(f"{self.name}: Started monitoring at {self.start_memory:.2f} GB")
        
    def measure(self, label=None, print_msg=False):
        """
        Take a measurement of current memory usage.
        
        Args:
            label: Label for this measurement (default: None)
            print_msg: Whether to print the measurement message (default: False)
        """
        if self.start_time is None:
            print(f"{self.name}: Monitoring not started. Call start() first.")
            return
        
        import time
        current_time = time.time() - self.start_time
        current_memory = t.cuda.memory_allocated() / (1024 * 1024 * 1024)  # GB
        memory_change = current_memory - self.start_memory
        
        # Use the label or create a default one
        if label is None:
            label = f"Measurement {len(self.measurements)}"
        
        self.measurements.append((current_time, current_memory, label))
        
        if print_msg:
            print(f"{self.name}: [{label}] Time: {current_time:.2f}s, Memory: {current_memory:.2f} GB, Change: {memory_change:+.2f} GB")
    
    def plot(self):
        """Plot memory usage over time."""
        import matplotlib.pyplot as plt
        
        if not self.measurements:
            print(f"{self.name}: No measurements to plot.")
            return
        
        times, memories, labels = zip(*self.measurements)
        
        plt.figure(figsize=(10, 6))
        plt.plot(times, memories, 'b-', marker='o')
        plt.title(f"{self.name} - GPU Memory Usage")
        plt.xlabel("Time (seconds)")
        plt.ylabel("Memory (GB)")
        plt.grid(True)
        plt.show()
    
    def reset(self):
        """Reset the monitor."""
        self.measurements = []
        self.start_time = None
        self.start_memory = None
        print(f"{self.name}: Reset")
        
    def start_continuous_monitoring(self, interval=10, print_msg=False):

        """
        Start continuous memory monitoring in a separate thread.
        
        Args:
            interval: Time between measurements in seconds (default: 10)
            print_msg: Whether to print a message when the monitoring starts (default: False)
        """
        import threading
        import time
        
        def monitor_loop():
            while not self._stop_monitoring:
                self.measure()
                time.sleep(interval)
        
        self._stop_monitoring = False
        self.monitor_thread = threading.Thread(target=monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        if print_msg:
            print(f"{self.name}: Started continuous monitoring every {interval} seconds")

    def stop_continuous_monitoring(self):
        """Stop the continuous monitoring thread."""
        self._stop_monitoring = True
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join()
            print(f"{self.name}: Stopped continuous monitoring")
    
    def report(self, return_history=False):
        """
        Generate a simple memory report showing measure statements and memory used.
        
        Args:
            return_history: Whether to return measurement history (default: False)
        
        Returns:
            If return_history=True: dict with measurement history and memory increase info
            Otherwise: None (prints formatted table to console)
        """
        try:
            from rich.console import Console
            from rich.table import Table
            from rich import box
            console = Console()
        except ImportError:
            print("Rich package not installed. Install with: pip install rich")
            if return_history:
                return None
            return
        
        # Create simple table
        table = Table(title=f"{self.name} - Memory Report", box=box.ROUNDED)
        table.add_column("Measure Statement")
        table.add_column("Memory Used (GB)")
        
        # Prepare measurement history for return
        history_data = []
        
        # Add measurement history
        if self.measurements:
            for i, (time_elapsed, memory_used, label) in enumerate(self.measurements):
                if label == "Start":
                    measure_statement = f"Start ({time_elapsed:.1f}s)"
                    memory_increase = 0.0  # Start measurement has no increase
                else:
                    measure_statement = f"{label} ({time_elapsed:.1f}s)"
                    # Calculate memory increase since last measurement
                    if i > 0:
                        memory_increase = memory_used - self.measurements[i-1][1]
                    else:
                        memory_increase = memory_used - self.start_memory
                
                table.add_row(measure_statement, f"{memory_used:.2f}")
                
                if return_history:
                    history_data.append({
                        'time': time_elapsed,
                        'description': label,
                        'memory_used_gb': memory_used,
                        'memory_increase_gb': memory_increase
                    })
        
        # Current GPU memory
        if t.cuda.is_available():
            gpu_memory_allocated = t.cuda.memory_allocated() / 1024**3  # GB
            table.add_row("Current GPU Memory", f"{gpu_memory_allocated:.2f}")
        
        # Current CPU memory
        cpu_memory_used = psutil.virtual_memory().used / 1024**3  # GB
        table.add_row("Current CPU Memory", f"{cpu_memory_used:.2f}")
        
        # Print the table
        console.print(table)
        
        # Return history if requested
        if return_history:
            return {
                'measurements': history_data,
                'current_gpu_memory_gb': gpu_memory_allocated if t.cuda.is_available() else None,
                'current_cpu_memory_gb': cpu_memory_used
            }

def clear_memory(variables_to_keep=None, clear_tensors=True, clear_cache=True):
    """
    Clears variables and releases GPU memory.
    
    Args:
        variables_to_keep: List of variable names to keep (default: None, keeps all)
        clear_tensors: Whether to clear PyTorch tensors (default: True)
        clear_cache: Whether to clear PyTorch cache (default: True)
    
    Returns:
        None
    """
    import gc
    
    # If no specific variables to keep, use an empty list
    if variables_to_keep is None:
        variables_to_keep = []
    
    # Get all variables in the current namespace
    current_vars = dict(locals())
    
    # Clear PyTorch cache
    if clear_cache and 't' in current_vars:
        t.cuda.empty_cache()
        t.cuda.reset_peak_memory_stats()
    
    # Run garbage collection
    gc.collect()
    
    # Clear variables that are not in the keep list
    for var_name, var in current_vars.items():
        # Skip internal variables and variables to keep
        if var_name.startswith('_') or var_name in variables_to_keep:
            continue
        
        # Delete the variable
        if var_name in locals():
            del locals()[var_name]
    
    # Print memory usage after clearing
    print_gpu_memory("after clearing memory")
    gc.collect()
    return None

def print_memory_usage():
    # Dictionary to store memory by type
    memory_by_type = defaultdict(int)

    # Create a copy of the locals dictionary to avoid the "dictionary changed size during iteration" error
    locals_copy = dict(locals())

    # Iterate through all variables in the current namespace
    for name, obj in locals_copy.items():
        # Skip internal/system variables
        if name.startswith('_'):
            continue
            
        # Get size in MB
        size_mb = sys.getsizeof(obj) / (1024 * 1024)
        
        # For PyTorch tensors, get actual memory allocation
        if isinstance(obj, t.Tensor):
            size_mb = obj.element_size() * obj.nelement() / (1024 * 1024)
            
        # For datasets, estimate size
        if str(type(obj).__module__).startswith('datasets'):
            size_mb = sum(sys.getsizeof(x) for x in obj.values()) / (1024 * 1024)
        
        type_name = type(obj).__name__
        memory_by_type[type_name] += size_mb
        
        print(f"Variable: {name:<20} Type: {type_name:<20} Size: {size_mb:.2f} MB")

    print("\nTotal memory by type:")
    for type_name, total_mb in memory_by_type.items():
        print(f"{type_name:<20}: {total_mb:.2f} MB")


# ----- steering utilities -----

def random_steering(
    target_vector: Float[Tensor, "... d_model"],
    original_vector: Float[Tensor, "... d_model"],
    steering_strength: float = 0.1,
):
    random_vector = t.randn_like(original_vector)
    # Compute norms along the d_model dimension only
    random_norm = t.norm(random_vector, dim=-1, keepdim=True)
    original_norm = t.norm(original_vector, dim=-1, keepdim=True)
    
    # Scale random vectors to match original magnitudes per position
    random_vector = random_vector / random_norm * original_norm
    
    new_vector = original_vector + steering_strength * (random_vector - original_vector)
    return new_vector


def interpolation_steering(
    target_vector: Float[Tensor, "... d_model"],
    original_vector: Float[Tensor, "... d_model"],
    steering_strength: float = 0.1,
):
    """
    Note:
        This method provides more controlled steering compared to direct_steering
        because the result is always a convex combination of the input vectors
        when steering_strength ≤ 1. This prevents the output from having
        unexpectedly large magnitudes.
    """
    if steering_strength > 1:
        raise ValueError("Steering strength must be less than 1")
    
    new_vector = original_vector + steering_strength * (target_vector - original_vector)
    return new_vector
    
