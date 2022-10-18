"""
Generate the gadfly logo!
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

np.random.seed(42)

# Set the maximum number of random attempts to draw non-overlapping
# "buds" before giving up:
max_draw_attempts = 1000

# Target number of "buds" to draw:
n_circles = 6

# Radius of each "bud"
radius = 0.05

# Save the logo here: 
logo_dir = os.path.dirname(__file__)
uncropped_svg_path = os.path.join(logo_dir, 'logo_uncropped.svg')
cropped_svg_path = os.path.join(logo_dir, 'logo.svg')
png_path = os.path.join(logo_dir, 'logo.png')
ico_path = os.path.join(logo_dir, 'logo.ico')

def no_overlaps(proposed_center, centers, exclusion_radius=2.2 * radius):
    """
    Returns True if the proposed circle center does not overlap with any
    existing circles.
    """
    for center in centers: 
        close_x = abs(proposed_center[0] - center[0]) < exclusion_radius
        close_y = abs(proposed_center[1] - center[1]) < exclusion_radius
        if close_x and close_y:
            return False
    return True

draw_attempts = 0
centers = []
fig, ax = plt.subplots(figsize=(2, 2))

# plot a randomly generated set of non-overlapping "hemlock buds":
while len(centers) < n_circles and draw_attempts < max_draw_attempts:
    r = np.random.uniform(0.01, 0.2)
    theta = np.random.uniform(0, 2*np.pi)
    proposed_center = [
        r * np.cos(theta),
        r * np.sin(theta)
    ]
    
    if no_overlaps(proposed_center, centers):
        centers.append(proposed_center)

        ax.add_patch(
            plt.Circle(proposed_center, radius, color='k')
        )
    draw_attempts += 1

if draw_attempts == max_draw_attempts:
    print(f"Stopping after {max_draw_attempts} bud placements.")
    
# connect the buds to a central "stem":
stem_x = 0
stem_y = -0.15

for x, y in centers:
    plt.plot([stem_x, x], [stem_y, y], lw=0.5, color='gray', zorder=0)
    
ax.set(
    xlim=[-0.4, 0.4], 
    ylim=[-0.4, 0.4]
)
ax.axis('off')

savefig_kwargs = dict(
    pad_inches=0, transparent=True
)

fig.savefig(uncropped_svg_path, **savefig_kwargs)

# PNG will be at *high* resolution:
fig.savefig(png_path, dpi=800, **savefig_kwargs)

# This is the default matplotlib SVG configuration which can't be easily tweaked:
default_svg_dims = 'width="144pt" height="144pt" viewBox="0 0 144 144"'

# This is a hand-tuned revision to the SVG file that crops the bounds nicely:
custom_svg_dims = 'width="75pt" height="75pt" viewBox="31 37 75 75"'

# Read the uncropped file, replace the bad configuration with the custom one:
with open(uncropped_svg_path, 'r') as svg:
    cropped_svg_source = svg.read().replace(
        default_svg_dims, custom_svg_dims
    )

# Write out the cropped SVG file:
with open(cropped_svg_path, 'w') as cropped_svg:
    cropped_svg.write(cropped_svg_source)

# Delete the uncropped SVG:
os.remove(uncropped_svg_path)

# Convert the PNG into an ICO file:
img = Image.open(png_path)
img.save(ico_path)
