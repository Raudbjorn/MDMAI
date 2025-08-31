#!/usr/bin/env python3
"""
Icon generation script for TTRPG Assistant Desktop
Generates all required icon sizes from a single high-resolution source image
"""

import os
import sys
from pathlib import Path
from typing import List, Tuple

try:
    from PIL import Image
except ImportError:
    print("Error: Pillow library is required. Install with: pip install Pillow")
    sys.exit(1)


def generate_windows_icons(source_image_path: str, output_dir: str) -> None:
    """
    Generate all required icon sizes for Windows/Tauri from a single source image
    
    Args:
        source_image_path: Path to source image (should be at least 1024x1024)
        output_dir: Directory to save generated icons
    """
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    (output_path / "tray").mkdir(exist_ok=True)
    
    try:
        # Open and validate source image
        source = Image.open(source_image_path).convert("RGBA")
        
        if source.width < 512 or source.height < 512:
            print("Warning: Source image should be at least 512x512 for best quality")
        
        print(f"Source image: {source.width}x{source.height}")
        
        # Standard icon sizes for Tauri
        standard_sizes = [
            (16, "16x16.png"),
            (24, "24x24.png"),
            (32, "32x32.png"),
            (48, "48x48.png"),
            (64, "64x64.png"),
            (128, "128x128.png"),
            (256, "128x128@2x.png"),  # Tauri naming convention for 2x
            (512, "icon.png"),  # Source icon
        ]
        
        # Windows Store tile sizes (optional but good to have)
        tile_sizes = [
            (30, "Square30x30Logo.png"),
            (44, "Square44x44Logo.png"),
            (71, "Square71x71Logo.png"),
            (89, "Square89x89Logo.png"),
            (107, "Square107x107Logo.png"),
            (142, "Square142x142Logo.png"),
            (150, "Square150x150Logo.png"),
            (284, "Square284x284Logo.png"),
            (310, "Square310x310Logo.png"),
        ]
        
        # Generate standard icons
        print("\nGenerating standard icons...")
        for size, filename in standard_sizes:
            img = source.resize((size, size), Image.Resampling.LANCZOS)
            output_file = output_path / filename
            img.save(output_file, "PNG", optimize=True)
            print(f"  ✓ {filename} ({size}x{size})")
        
        # Generate tile icons
        print("\nGenerating Windows Store tiles...")
        for size, filename in tile_sizes:
            img = source.resize((size, size), Image.Resampling.LANCZOS)
            
            # Add padding for better tile appearance
            padding = int(size * 0.1)  # 10% padding
            canvas_size = size
            actual_size = size - (padding * 2)
            
            if actual_size > 0:
                img_resized = source.resize((actual_size, actual_size), Image.Resampling.LANCZOS)
                canvas = Image.new('RGBA', (canvas_size, canvas_size), (0, 0, 0, 0))
                offset = padding
                canvas.paste(img_resized, (offset, offset))
                output_file = output_path / filename
                canvas.save(output_file, "PNG", optimize=True)
            else:
                output_file = output_path / filename
                img.save(output_file, "PNG", optimize=True)
            
            print(f"  ✓ {filename} ({size}x{size})")
        
        # Generate multi-size ICO file for Windows
        print("\nGenerating Windows ICO file...")
        ico_sizes = [(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)]
        ico_images = []
        
        for size in ico_sizes:
            img = source.resize(size, Image.Resampling.LANCZOS)
            ico_images.append(img)
        
        ico_output = output_path / "icon.ico"
        ico_images[0].save(
            ico_output,
            format="ICO",
            sizes=ico_sizes,
            append_images=ico_images[1:]
        )
        print(f"  ✓ icon.ico (multi-size)")
        
        # Generate tray icons (can be customized later for different states)
        print("\nGenerating system tray icons...")
        tray_sizes = [16, 24, 32]
        
        for state in ["icon", "icon-active", "icon-error", "icon-syncing"]:
            # For now, use the same icon for all states
            # In production, you'd want different icons for each state
            tray_ico_images = []
            
            for size in tray_sizes:
                img = source.resize((size, size), Image.Resampling.LANCZOS)
                
                # Apply simple color tint for different states (placeholder)
                if state == "icon-active":
                    # Green tint for active
                    img = apply_color_overlay(img, (0, 255, 0, 50))
                elif state == "icon-error":
                    # Red tint for error
                    img = apply_color_overlay(img, (255, 0, 0, 50))
                elif state == "icon-syncing":
                    # Blue tint for syncing
                    img = apply_color_overlay(img, (0, 100, 255, 50))
                
                tray_ico_images.append(img)
            
            tray_output = output_path / "tray" / f"{state}.ico"
            tray_ico_images[0].save(
                tray_output,
                format="ICO",
                sizes=[(s, s) for s in tray_sizes],
                append_images=tray_ico_images[1:]
            )
            print(f"  ✓ tray/{state}.ico")
        
        # Generate macOS ICNS file (if on macOS)
        try:
            import subprocess
            if sys.platform == "darwin":
                print("\nGenerating macOS ICNS file...")
                icns_output = output_path / "icon.icns"
                # This requires iconutil on macOS
                iconset_path = output_path / "icon.iconset"
                iconset_path.mkdir(exist_ok=True)
                
                mac_sizes = [
                    (16, "16x16"),
                    (32, "16x16@2x"),
                    (32, "32x32"),
                    (64, "32x32@2x"),
                    (128, "128x128"),
                    (256, "128x128@2x"),
                    (256, "256x256"),
                    (512, "256x256@2x"),
                    (512, "512x512"),
                    (1024, "512x512@2x"),
                ]
                
                for size, name in mac_sizes:
                    img = source.resize((size, size), Image.Resampling.LANCZOS)
                    img.save(iconset_path / f"icon_{name}.png", "PNG")
                
                subprocess.run(["iconutil", "-c", "icns", str(iconset_path), "-o", str(icns_output)])
                print(f"  ✓ icon.icns")
                
                # Clean up iconset
                import shutil
                shutil.rmtree(iconset_path)
        except Exception as e:
            print(f"  ⚠ Could not generate ICNS file: {e}")
        
        print(f"\n✅ Successfully generated all icons in: {output_path}")
        
    except Exception as e:
        print(f"\n❌ Error generating icons: {e}")
        sys.exit(1)


def apply_color_overlay(image: Image.Image, color: Tuple[int, int, int, int]) -> Image.Image:
    """Apply a color overlay to an image (for tray icon states)"""
    overlay = Image.new('RGBA', image.size, color)
    return Image.alpha_composite(image, overlay)


def create_placeholder_logo(output_path: str) -> None:
    """Create a placeholder logo if no source image exists"""
    print("Creating placeholder logo...")
    
    # Create a simple gradient logo as placeholder
    size = 1024
    img = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    
    # Draw a simple dice icon as placeholder (for TTRPG theme)
    from PIL import ImageDraw, ImageFont
    
    draw = ImageDraw.Draw(img)
    
    # Background gradient
    for i in range(size):
        alpha = int(255 * (1 - i / size * 0.3))
        color = (102, 126, 234, alpha)  # Purple-blue gradient
        draw.rectangle([0, i, size, i+1], fill=color)
    
    # Draw a simple D20 shape
    center = size // 2
    radius = size // 3
    
    # Draw hexagon (simplified D20 face)
    import math
    points = []
    for i in range(6):
        angle = math.radians(60 * i)
        x = center + radius * math.cos(angle)
        y = center + radius * math.sin(angle)
        points.append((x, y))
    
    draw.polygon(points, fill=(255, 255, 255, 200), outline=(255, 255, 255, 255), width=8)
    
    # Add text
    try:
        # Try to use a nice font if available
        font = ImageFont.truetype("arial.ttf", size // 4)
    except (OSError, IOError, ImportError):
        font = ImageFont.load_default()
    
    text = "d20"
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    text_x = center - text_width // 2
    text_y = center - text_height // 2
    
    draw.text((text_x, text_y), text, fill=(100, 75, 162, 255), font=font)
    
    # Save placeholder
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path, "PNG")
    print(f"  ✓ Created placeholder at: {output_path}")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate icons for TTRPG Assistant Desktop")
    parser.add_argument(
        "source",
        nargs="?",
        default="assets/logo-source.png",
        help="Path to source image (default: assets/logo-source.png)"
    )
    parser.add_argument(
        "-o", "--output",
        default="frontend/src-tauri/icons",
        help="Output directory (default: frontend/src-tauri/icons)"
    )
    parser.add_argument(
        "--create-placeholder",
        action="store_true",
        help="Create a placeholder logo if source doesn't exist"
    )
    
    args = parser.parse_args()
    
    # Check if source exists
    if not Path(args.source).exists():
        if args.create_placeholder:
            create_placeholder_logo(args.source)
        else:
            print(f"Error: Source image '{args.source}' not found.")
            print("Use --create-placeholder to generate a placeholder logo.")
            sys.exit(1)
    
    # Generate icons
    generate_windows_icons(args.source, args.output)


if __name__ == "__main__":
    main()