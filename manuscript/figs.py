import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

def generate_figure_1(output_path):
    """
    Figure 1: Execution Lifecycle
    Reduced width to fit paper boundaries. High readability.
    """
    fig = plt.figure(figsize=(9, 3)) # Reduced from 12
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 100) # Standardized scale
    ax.set_ylim(0, 40)
    ax.axis('off')

    nodes = [
        "Immutable IR",
        "Static Schedule",
        "Static Arena",
        "Kernel Dispatch",
        "Trace"
    ]
    
    # Professional Blue Palette
    box_colors = ['#E3F2FD', '#BBDEFB', '#90CAF9', '#64B5F6', '#42A5F5']
    border_color = '#1565C0'
    
    box_w = 16
    box_h = 14
    y_center = 20
    start_x = 2
    spacing = 4

    for i, text in enumerate(nodes):
        x = start_x + i * (box_w + spacing)
        
        # Rounded box
        rect = patches.FancyBboxPatch(
            (x, y_center - box_h/2), box_w, box_h,
            boxstyle="round,pad=0.2,rounding_size=1",
            linewidth=1.2, edgecolor=border_color, facecolor=box_colors[i]
        )
        ax.add_patch(rect)
        
        # Text
        ax.text(x + box_w/2, y_center, text.replace(" ", "\n"), 
                ha='center', va='center', fontsize=9, fontfamily='sans-serif', fontweight='bold', color='#0D47A1')
        
        # Arrow
        if i < len(nodes) - 1:
            ax.annotate("", 
                        xy=(x + box_w + spacing, y_center), 
                        xytext=(x + box_w, y_center),
                        arrowprops=dict(arrowstyle="->", color='#455A64', lw=1.5, mutation_scale=12))

    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

def generate_figure_2(output_path):
    """
    Figure 2: Trace / Provenance Model
    Fixed overlay issues. Sequential event list with clear separation.
    """
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')

    events = [
        "GraphCompilation",
        "MemoryAllocation",
        "KernelDispatch",
        "NodeExecutionStart",
        "NodeExecutionEnd"
    ]
    
    border_color = '#33691E'
    # High-contrast text color
    text_color = '#1B5E20'
    
    start_y = 90
    y_step = 18 # Increased spacing
    x_pos = 20
    box_w = 60
    box_h = 10

    # Vertical Timeline (Moved to left side of boxes to avoid overlay)
    timeline_x = x_pos - 5
    ax.plot([timeline_x, timeline_x], [5, 95], color='#BDBDBD', linestyle='-', linewidth=2, zorder=0)

    for i, text in enumerate(events):
        y = start_y - i * y_step
        
        # Marker on timeline
        ax.plot(timeline_x, y, marker='o', markersize=8, color=border_color, zorder=2)
        
        # Connection line from marker to box
        ax.plot([timeline_x, x_pos], [y, y], color='#BDBDBD', linestyle='-', linewidth=1, zorder=1)
        
        # Event Box
        rect = patches.FancyBboxPatch(
            (x_pos, y - box_h/2), box_w, box_h,
            boxstyle="round,pad=0.3",
            linewidth=1.0, edgecolor=border_color, facecolor='#F1F8E9', zorder=1
        )
        ax.add_patch(rect)
        
        # Text
        ax.text(x_pos + box_w/2, y, text, ha='center', va='center', 
                fontsize=10, fontfamily='sans-serif', fontweight='bold', color=text_color)

    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

if __name__ == "__main__":
    out_dir = "manuscript/figures"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    generate_figure_1(os.path.join(out_dir, "execution_lifecycle.png"))
    generate_figure_2(os.path.join(out_dir, "trace_model.png"))