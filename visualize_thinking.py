import torch as t
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import numpy as np
from typing import List, Tuple, Any

def visualize_thinking_heatmap(
    str_array: List[List[List[str]]],
    values: t.Tensor,
    layer_list: List[int],
    title: str = "Thinking Process Visualization",
    save_path: str = None
):
    """
    Create a heatmap visualization of the thinking process across layers.
    
    Args:
        str_array: 3D list of shape [seq_len, num_layers, k] containing token strings
        values: Tensor of shape [num_layers, seq_len, k] containing probability values
        layer_list: List of layer indices being visualized
        title: Title for the overall figure
        save_path: Optional path to save the HTML file
        
    Returns:
        plotly.graph_objects.Figure: The created figure
    """
    
    # Create subplots with different heights for first subplot
    # First subplot gets 0.3 of the height, others get 0.7/(num_layers-1) each
    num_layers = len(layer_list)
    if num_layers == 1:
        heights = [1.0]
    else:
        heights = [0.3] + [0.7 / (num_layers - 1)] * (num_layers - 1)
    
    fig = make_subplots(
        rows=num_layers, 
        cols=1,
        subplot_titles=[f'Layer {layer_idx}' for layer_idx in layer_list],
        vertical_spacing=0.05,
        row_heights=heights
    )

    for (i, layer_idx) in enumerate(layer_list):
        # Create heatmap for this layer
        layer_values = values[i]  # Shape: [seq_len, k]
        
        # Create heatmap with reversed y-axis (Top-1 at top)
        heatmap = go.Heatmap(
            z=layer_values.transpose(0, 1).detach().to(t.float32).numpy(),
            y=[f"Top-{3-j}" for j in range(3)],  # Reversed order: Top-3, Top-2, Top-1
            x=[f"Pos-{j}" for j in range(layer_values.shape[0])],
            colorscale=[[0, 'white'], [1, '#ffcccc']],  # White to light red
            name=f'Layer {layer_idx}',
            showscale=True if i == 0 else False,  # Only show colorbar for first layer
            colorbar=dict(title="Probability") if i == 0 else None
        )
        
        # Add text annotations for each cell with reversed y position
        for seq_pos in range(layer_values.shape[0]):
            for k_idx in range(3):
                token_text = str_array[seq_pos][i][k_idx]
                # Add text annotation with reversed y position
                fig.add_annotation(
                    x=seq_pos,
                    y=2-k_idx,  # Reversed: 2, 1, 0 instead of 0, 1, 2
                    text=token_text,
                    showarrow=False,
                    font=dict(size=15, color="black"),
                    xanchor='center',
                    yanchor='middle',
                    row=i+1,  # Add row parameter for subplot positioning
                    col=1
                )
        
        fig.add_trace(heatmap, row=i+1, col=1)
        
        # Configure axes for this subplot
        # Hide x-axis labels for all subplots except the last one
        if i < num_layers - 1:
            fig.update_xaxes(showticklabels=False, row=i+1, col=1)
        else:
            fig.update_xaxes(title_text="Sequence Position", row=i+1, col=1)
        
        # Configure y-axis
        fig.update_yaxes(title_text="Top-k Tokens", row=i+1, col=1)

    # Update layout
    fig.update_layout(
        title=title,
        height=200 * num_layers,  # Adjust height based on number of layers
        width=800,
        showlegend=False
    )
    
    # Save if path provided
    if save_path:
        fig.write_html(save_path)
        print(f"Figure saved to {save_path}")
    
    return fig

def load_thinking_data(model_name: str, k: int = 3, save_dir: str = "data/thinking") -> Tuple[List[List[List[str]]], t.Tensor, t.Tensor]:
    """
    Load the thinking data from saved files.
    
    Args:
        model_name: Name of the model
        k: Number of top-k tokens
        save_dir: Directory where data is saved
        
    Returns:
        Tuple of (str_array, values, indices)
    """
    # Load string data
    with open(f"{save_dir}/topk_decoder_{model_name}_top{k}_strings.json", "r") as f:
        str_array = json.load(f)
    
    # Load numeric data
    numeric_data = t.load(f"{save_dir}/topk_decoder_{model_name}_top{k}.pt")
    values = numeric_data["values"]
    indices = numeric_data["indices"]
    
    return str_array, values, indices

def create_thinking_visualization(
    model_name: str,
    layer_list: List[int],
    prompt: str,
    k: int = 3,
    save_dir: str = "data/thinking",
    output_path: str = None
) -> go.Figure:
    """
    Complete pipeline to create thinking visualization from scratch.
    
    Args:
        model_name: Name of the model to analyze
        layer_list: List of layer indices to visualize
        prompt: Input prompt
        k: Number of top-k tokens
        save_dir: Directory to save/load data
        output_path: Optional path to save the HTML visualization
        
    Returns:
        plotly.graph_objects.Figure: The created visualization
    """
    from thinking import topk_decoder
    
    # Generate the data
    str_array, values, indices = topk_decoder(
        model_name=model_name,
        layer_list=layer_list,
        device="cuda",
        prompt=prompt,
        k=k,
        save_dir=save_dir
    )
    
    # Create visualization
    title = f"Thinking Process: {prompt[:50]}{'...' if len(prompt) > 50 else ''}"
    fig = visualize_thinking_heatmap(
        str_array=str_array,
        values=values,
        layer_list=layer_list,
        title=title,
        save_path=output_path
    )
    
    return fig

if __name__ == "__main__":
    # Example usage
    import numpy as np
    
    # Option 1: Load existing data and visualize
    try:
        str_array, values, indices = load_thinking_data("gpt2-xl", k=3)
        layer_list = list(np.linspace(0, 47, 6, dtype=int))
        
        fig = visualize_thinking_heatmap(
            str_array=str_array,
            values=values,
            layer_list=layer_list,
            title="GPT-2 XL Thinking Process",
            save_path="thinking_visualization.html"
        )
        fig.show()
        
    except FileNotFoundError:
        print("No existing data found. Run the complete pipeline instead.")
        
        # Option 2: Generate data and visualize
        fig = create_thinking_visualization(
            model_name="gpt2-xl",
            layer_list=list(np.linspace(0, 47, 6, dtype=int)),
            prompt="The cat sat on the mat.",
            k=3,
            output_path="thinking_visualization.html"
        )
        fig.show() 