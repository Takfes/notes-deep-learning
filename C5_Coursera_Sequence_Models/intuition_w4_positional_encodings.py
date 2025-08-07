import numpy as np
import matplotlib.pyplot as plt


def get_angles(pos, k, d):
    """
    Get the angles for the positional encoding

    Arguments:
        pos -- Column vector containing the positions [[0], [1], ...,[N-1]]
        k --   Row vector containing the dimension span [[0, 1, 2, ..., d-1]]
        d(integer) -- Encoding size

    Returns:
        angles -- (pos, d) numpy array
    """

    # START CODE HERE
    # Get i from dimension span k
    i = k // 2
    # Calculate the angles using pos, i and d
    angles = pos / (10_000 ** (2 * i / d))
    # END CODE HERE

    return angles


def positional_encoding(positions, d):
    """
    Precomputes a matrix with all the positional encodings

    Arguments:
        positions (int) -- Maximum number of positions to be encoded
        d (int) -- Encoding size

    Returns:
        pos_encoding -- (1, position, d_model) A matrix with the positional encodings
    """
    # START CODE HERE
    # initialize a matrix angle_rads of all the angles
    angle_rads = get_angles(
        np.arange(positions)[:, np.newaxis], np.arange(d)[np.newaxis, :], d
    )

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    # END CODE HERE

    pos_encoding = angle_rads[np.newaxis, ...]

    return pos_encoding.astype(np.float32)


def demo_positional_encodings():
    """Demonstrate positional encodings with visualizations and examples."""

    print("=" * 60)
    print("POSITIONAL ENCODINGS INTUITION")
    print("=" * 60)

    # Parameters
    max_positions = 50
    d_model = 128

    print("\nGenerating positional encodings for:")
    print(f"- Maximum positions: {max_positions}")
    print(f"- Model dimension: {d_model}")

    # Generate positional encodings
    pos_encoding = positional_encoding(max_positions, d_model)
    print(f"- Shape of positional encoding matrix: {pos_encoding.shape}")

    # Remove the batch dimension for visualization
    pos_enc = pos_encoding[0]

    print("\n" + "=" * 60)
    print("UNDERSTANDING THE ARGUMENTS")
    print("=" * 60)

    print("\n1. POSITION (pos):")
    print("   - Represents the position of a token in the sequence")
    print("   - For sequence 'Hello world !', positions would be [0, 1, 2]")
    print("   - Each position gets a unique encoding pattern")

    print("\n2. DIMENSION INDEX (k):")
    print("   - Each dimension of the embedding gets a different frequency")
    print("   - Lower dimensions change slowly, higher dimensions change quickly")
    print("   - This creates a unique 'fingerprint' for each position")

    print("\n3. MODEL DIMENSION (d):")
    print("   - Total size of the embedding (e.g., 512 in original Transformer)")
    print("   - Determines how many different frequencies we use")
    print("   - More dimensions = more precise position information")

    print("\n" + "=" * 60)
    print("VISUALIZATION EXAMPLES")
    print("=" * 60)

    # Show first few positions and dimensions
    print("\nFirst 5 positions, first 8 dimensions:")
    print(pos_enc[:5, :8])

    print("\nNotice how:")
    print("- Each row (position) has a unique pattern")
    print("- Even columns use sin, odd columns use cos")
    print("- Values are between -1 and 1")

    # Show how different positions have different patterns
    print("\nComparing positions 0, 10, and 20 (first 8 dims):")
    for pos in [0, 10, 20]:
        print(f"Position {pos:2d}: {pos_enc[pos, :8]}")

    print("\nKey insights:")
    print("- Position 0 starts with specific pattern")
    print("- Each position has a mathematically unique encoding")
    print("- The model can learn to use these patterns for attention")

    # Demonstrate the frequency concept
    print("\n" + "=" * 60)
    print("FREQUENCY PATTERNS")
    print("=" * 60)

    # Show how different dimensions have different frequencies
    plt.figure(figsize=(15, 10))

    # Plot first few dimensions across positions
    plt.subplot(2, 2, 1)
    for dim in [0, 2, 4, 6]:  # Even dimensions (sin)
        plt.plot(pos_enc[:30, dim], label=f"Dim {dim} (sin)")
    plt.title("Even Dimensions (Sin Functions)")
    plt.xlabel("Position")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 2)
    for dim in [1, 3, 5, 7]:  # Odd dimensions (cos)
        plt.plot(pos_enc[:30, dim], label=f"Dim {dim} (cos)")
    plt.title("Odd Dimensions (Cos Functions)")
    plt.xlabel("Position")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)

    # Heatmap of positional encodings
    plt.subplot(2, 1, 2)
    plt.imshow(pos_enc[:30, :50].T, cmap="RdBu", aspect="auto")
    plt.colorbar(label="Encoding Value")
    plt.title("Positional Encoding Heatmap (30 positions × 50 dimensions)")
    plt.xlabel("Position")
    plt.ylabel("Dimension")

    plt.tight_layout()
    plt.savefig(
        "/Users/takis/Documents/sckool/notes-deep-learning/positional_encoding_visualization.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()

    print("\n" + "=" * 60)
    print("WHY THIS DESIGN WORKS")
    print("=" * 60)

    print("\n1. UNIQUENESS:")
    print("   - Each position gets a unique combination of sin/cos values")
    print("   - No two positions have identical encodings")

    print("\n2. RELATIVE POSITIONS:")
    print("   - The model can learn relative distances between positions")
    print("   - Mathematical properties allow computing PE(pos+k) from PE(pos)")

    print("\n3. EXTRAPOLATION:")
    print("   - Can handle sequences longer than seen during training")
    print("   - The mathematical formula extends naturally")

    print("\n4. SMOOTH TRANSITIONS:")
    print("   - Nearby positions have similar encodings")
    print("   - Gradual changes help the model generalize")

    # Show similarity between nearby positions
    print("\nSimilarity example (cosine similarity):")
    pos_0 = pos_enc[0]
    pos_1 = pos_enc[1]
    pos_10 = pos_enc[10]

    sim_0_1 = np.dot(pos_0, pos_1) / (np.linalg.norm(pos_0) * np.linalg.norm(pos_1))
    sim_0_10 = np.dot(pos_0, pos_10) / (np.linalg.norm(pos_0) * np.linalg.norm(pos_10))

    print(f"- Similarity between pos 0 and pos 1:  {sim_0_1:.4f}")
    print(f"- Similarity between pos 0 and pos 10: {sim_0_10:.4f}")
    print("- Nearby positions are more similar!")


def detailed_breakdown_example():
    """Step-by-step breakdown of how positional encodings work"""

    print("=" * 80)
    print("DETAILED POSITIONAL ENCODING BREAKDOWN")
    print("=" * 80)

    # Simple example
    sentence = ["Hello", "world", "!"]
    d_model = 8  # Small dimension for clarity

    print(f"\nExample sentence: {sentence}")
    print(f"Model dimension (d): {d_model}")
    print(f"Number of positions: {len(sentence)}")

    print("\n" + "=" * 80)
    print("STEP 1: UNDERSTANDING THE DIMENSIONS")
    print("=" * 80)

    print(f"\nFor each word, we create a {d_model}-dimensional vector:")
    print("- Dimension indices (k): [0, 1, 2, 3, 4, 5, 6, 7]")
    print("- Even dimensions (sin): [0, 2, 4, 6]  → use sin function")
    print("- Odd dimensions (cos):  [1, 3, 5, 7]  → use cos function")

    print("\n" + "=" * 80)
    print("STEP 2: THE i PARAMETER")
    print("=" * 80)

    print("\nThe i parameter groups dimension pairs:")
    for k in range(d_model):
        i = k // 2
        func = "sin" if k % 2 == 0 else "cos"
        print(f"k={k} → i={i} → {func}(pos / 10000^(2*{i}/{d_model}))")

    print("\n" + "=" * 80)
    print("STEP 3: COMPUTING ACTUAL VALUES")
    print("=" * 80)

    # Generate the actual positional encodings
    pos_encoding = positional_encoding(len(sentence), d_model)
    pos_enc = pos_encoding[0]  # Remove batch dimension

    print(f"\nActual positional encoding matrix shape: {pos_enc.shape}")
    print("(rows = positions, columns = dimensions)")

    for pos, word in enumerate(sentence):
        print(f"\n'{word}' (position {pos}):")
        print(f"  Full vector: {pos_enc[pos]}")

        # Show the calculation for first few dimensions
        print("  Breakdown:")
        for k in range(min(4, d_model)):
            i = k // 2
            angle = pos / (10000 ** (2 * i / d_model))
            if k % 2 == 0:  # Even - sin
                value = np.sin(angle)
                print(
                    f"    dim {k} (sin): sin({pos} / 10000^({2*i}/{d_model})) = sin({angle:.6f}) = {value:.6f}"
                )
            else:  # Odd - cos
                value = np.cos(angle)
                print(
                    f"    dim {k} (cos): cos({pos} / 10000^({2*i}/{d_model})) = cos({angle:.6f}) = {value:.6f}"
                )

    print("\n" + "=" * 80)
    print("STEP 4: WHY SIN/COS PAIRING WORKS")
    print("=" * 80)

    print("\n1. FREQUENCY SEPARATION:")
    print("   Each i creates a different frequency:")
    for i in range(4):
        freq = 1 / (10000 ** (2 * i / d_model))
        print(f"   i={i}: frequency = {freq:.8f}")

    print("\n2. COMPLEMENTARY INFORMATION:")
    print("   Sin and cos provide orthogonal signals:")

    # Show sin/cos relationship for position 1
    pos = 1
    for i in range(2):
        angle = pos / (10000 ** (2 * i / d_model))
        sin_val = np.sin(angle)
        cos_val = np.cos(angle)
        print(f"   i={i}: sin={sin_val:.4f}, cos={cos_val:.4f}")
        print(f"        sin²+cos² = {sin_val**2 + cos_val**2:.4f} (always 1)")

    print("\n3. UNIQUE PATTERNS:")
    print("   Each position gets a unique fingerprint:")

    # Show how positions differ
    print("\n   Position comparison (first 4 dimensions):")
    for pos in range(len(sentence)):
        print(f"   pos {pos}: {pos_enc[pos, :4]}")

    print("\n" + "=" * 80)
    print("STEP 5: VISUAL INTUITION")
    print("=" * 80)

    # Create a more detailed visualization
    plt.figure(figsize=(16, 12))

    # Plot 1: Show the angle calculation
    positions = np.arange(20)
    plt.subplot(3, 2, 1)
    for i in [0, 1, 2]:
        angles = positions / (10000 ** (2 * i / d_model))
        plt.plot(
            positions, angles, label=f"i={i}: pos/10000^({2*i}/{d_model})", marker="o"
        )
    plt.title("Angle Calculation: pos / 10000^(2i/d)")
    plt.xlabel("Position")
    plt.ylabel("Angle (radians)")
    plt.legend()
    plt.grid(True)

    # Plot 2: Sin functions
    plt.subplot(3, 2, 2)
    for i in [0, 1, 2]:
        angles = positions / (10000 ** (2 * i / d_model))
        sin_vals = np.sin(angles)
        plt.plot(positions, sin_vals, label=f"dim {2*i} (sin, i={i})", marker="o")
    plt.title("Even Dimensions (Sin Functions)")
    plt.xlabel("Position")
    plt.ylabel("Sin Value")
    plt.legend()
    plt.grid(True)

    # Plot 3: Cos functions
    plt.subplot(3, 2, 3)
    for i in [0, 1, 2]:
        angles = positions / (10000 ** (2 * i / d_model))
        cos_vals = np.cos(angles)
        plt.plot(positions, cos_vals, label=f"dim {2*i+1} (cos, i={i})", marker="o")
    plt.title("Odd Dimensions (Cos Functions)")
    plt.xlabel("Position")
    plt.ylabel("Cos Value")
    plt.legend()
    plt.grid(True)

    # Plot 4: 2D visualization of sin/cos pairs
    plt.subplot(3, 2, 4)
    for i in [0, 1]:
        angles = positions[:10] / (10000 ** (2 * i / d_model))
        sin_vals = np.sin(angles)
        cos_vals = np.cos(angles)
        plt.plot(sin_vals, cos_vals, "o-", label=f"i={i} (sin,cos pairs)")
        # Add position labels
        for pos, (s, c) in enumerate(zip(sin_vals, cos_vals)):
            if pos % 2 == 0:  # Label every other point
                plt.annotate(
                    f"{pos}",
                    (s, c),
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontsize=8,
                )
    plt.title("Sin/Cos Pairs for Different i values")
    plt.xlabel("Sin Value")
    plt.ylabel("Cos Value")
    plt.legend()
    plt.grid(True)
    plt.axis("equal")

    # Plot 5: Full encoding heatmap
    plt.subplot(3, 1, 3)
    larger_encoding = positional_encoding(20, 16)[0]
    plt.imshow(larger_encoding.T, cmap="RdBu", aspect="auto")
    plt.colorbar(label="Encoding Value")
    plt.title("Complete Positional Encoding Heatmap")
    plt.xlabel("Position")
    plt.ylabel("Dimension")

    # Add lines to show sin/cos alternation
    for k in range(16):
        color = "white" if k % 2 == 0 else "yellow"
        alpha = 0.3 if k % 2 == 0 else 0.2
        plt.axhline(y=k - 0.5, color=color, alpha=alpha, linewidth=1)

    plt.tight_layout()
    plt.savefig(
        "/Users/takis/Documents/sckool/notes-deep-learning/detailed_positional_encoding.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()

    print("\nKey Takeaways:")
    print("1. Each word gets a d-dimensional positional vector")
    print("2. Even dimensions use sin, odd dimensions use cos")
    print("3. Different i values create different frequencies")
    print("4. Sin/cos pairs provide complementary information")
    print("5. Each position has a unique mathematical fingerprint")


if __name__ == "__main__":
    detailed_breakdown_example()
    print("\n" + "=" * 80 + "\n")
    demo_positional_encodings()
