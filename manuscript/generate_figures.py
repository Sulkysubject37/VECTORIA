import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

def generate_figure_1(output_path):
    """
    Figure 1: VECTORIA Execution Lifecycle
    Professional aesthetic with high contrast.
    """
    border_color = '#0D47A1'
    fill_color = '#E3F2FD'
    text_color = '#000000'
    arrow_color = '#424242'

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 40)
    ax.axis('off')

    nodes = [
        "Immutable IR",
        "Static Schedule",
        "Static Arena",
        "Kernel Dispatch",
        "Trace"
    ]
    
    node_width = 16
    node_height = 10
    spacing = 4
    start_x = 2
    y_pos = 15

    for i, node in enumerate(nodes):
        x = start_x + i * (node_width + spacing)
        
        # Draw Box
        rect = patches.FancyBboxPatch(
            (x, y_pos), node_width, node_height,
            boxstyle="round,pad=0.3,rounding_size=1",
            linewidth=1.5, edgecolor=border_color, facecolor=fill_color
        )
        ax.add_patch(rect)
        
        # Add Text
        ax.text(
            x + node_width/2, y_pos + node_height/2, node.replace(" ", "\n"),
            ha='center', va='center', fontfamily='sans-serif', fontsize=10, 
            fontweight='bold', color=text_color
        )
        
        # Draw Arrow
        if i < len(nodes) - 1:
            ax.annotate("", 
                        xy=(x + node_width + spacing, y_pos + node_height/2), 
                        xytext=(x + node_width, y_pos + node_height/2),
                        arrowprops=dict(arrowstyle="->", color=arrow_color, lw=2, mutation_scale=15))

    plt.title("Figure 1: VECTORIA Deterministic Execution Lifecycle", 
              fontsize=14, fontweight='bold', pad=25, color=border_color)
    
    plt.savefig(output_path, format='svg', bbox_inches='tight')
    plt.close()

def generate_figure_2(output_path):
    """
    Figure 2: Trace & Provenance Model
    High-readability sequential event log.
    """
    border_color = '#1B5E20'
    fill_color = '#E8F5E9'
    text_color = '#000000'
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')

    events = [
        {"type": "MemoryAllocation", "desc": "Node 0 | 1024 Bytes"},
        {"type": "NodeExecutionStart", "desc": "Node 2 (MatMul)"},
        {"type": "KernelDispatch", "desc": "Node 2 | SIMD-AVX2"},
        {"type": "NodeExecutionEnd", "desc": "Node 2 Complete"}
    ]
    
    start_y = 80
    y_step = 20
    x_pos = 10

    # Draw vertical timeline
    ax.plot([x_pos, x_pos], [10, 85], color='#BDBDBD', linestyle='-', linewidth=2)

    for i, event in enumerate(events):
        y = start_y - i * y_step
        
        # Marker
        ax.plot(x_pos, y, marker='o', markersize=8, color=border_color, markeredgecolor='white', markeredgewidth=1.5)
        
        # Label Box
        rect = patches.FancyBboxPatch(
            (x_pos + 5, y - 6), 75, 12,
            boxstyle="round,pad=0.2",
            linewidth=1, edgecolor=border_color, facecolor=fill_color
        )
        ax.add_patch(rect)
        
        # Text
        label = f"{event['type']}: {event['desc']}"
        ax.text(
            x_pos + 7, y, label,
            ha='left', va='center', fontfamily='monospace', fontsize=10, color=text_color
        )

    plt.title("Figure 2: Execution Tracing and Scientific Provenance", 
              fontsize=14, fontweight='bold', pad=25, color=border_color)
    
    plt.savefig(output_path, format='svg', bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    output_dir = "manuscript/figures"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    generate_figure_1(os.path.join(output_dir, "figure1_lifecycle.svg"))
    generate_figure_2(os.path.join(output_dir, "figure2_trace_model.svg"))
    print("Figures regenerated successfully.")
