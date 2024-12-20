import cv2
import numpy as np
import random
import os 

def remove_white_background(image_path):
    """Remove white background and extract the contour of the puzzle piece."""
    # Load image
    image_path = 'pieces/' + image_path
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Threshold to create a binary mask
    _, binary = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)

    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Assume the largest contour is the puzzle piece
    contour = max(contours, key=cv2.contourArea)

    # Create a mask for the puzzle piece
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)

    # Extract the puzzle piece using the mask
    piece = cv2.bitwise_and(image, image, mask=mask)

    return piece, contour

def extract_edges_from_contour(contour):
    """Split the contour into four edges: top, bottom, left, right."""
    # Get the bounding box for the puzzle piece
    x, y, w, h = cv2.boundingRect(contour)

    # Approximate contour points to simplify edge extraction
    epsilon = 0.01 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)

    # Initialize edges
    top_edge = []
    bottom_edge = []
    left_edge = []
    right_edge = []

    for point in approx:
        px, py = point[0]
        if py < y + h * 0.25:
            top_edge.append((px, py))
        elif py > y + h * 0.75:
            bottom_edge.append((px, py))
        elif px < x + w * 0.25:
            left_edge.append((px, py))
        elif px > x + w * 0.75:
            right_edge.append((px, py))

    return {
        "top": np.array(top_edge),
        "bottom": np.array(bottom_edge),
        "left": np.array(left_edge),
        "right": np.array(right_edge),
    }

def calculate_edge_similarity(edge1, edge2):
    """Calculate similarity between two edges using shape matching."""
    # Use Hausdorff distance or other contour comparison metrics
    score = cv2.matchShapes(edge1, edge2, cv2.CONTOURS_MATCH_I1, 0)
    return score

def find_best_matches(all_edges):
    """Find the best matching edge for each piece."""
    matches = {}
    for i, piece1 in enumerate(all_edges):
        for edge1_name, edge1 in piece1['edges'].items():
            best_match = None
            best_similarity = float('inf')
            for j, piece2 in enumerate(all_edges):
                if i != j:  # Avoid comparing the same piece
                    for edge2_name, edge2 in piece2['edges'].items():
                        similarity = calculate_edge_similarity(edge1, edge2)
                        if similarity < best_similarity:
                            best_similarity = similarity
                            best_match = (j, edge2_name)
            matches[(i, edge1_name)] = best_match
    return matches

def initialize_population(pieces, population_size=10):
    """Generate initial population with random arrangements."""
    population = []
    for _ in range(population_size):
        arrangement = random.sample(pieces, len(pieces))
        population.append(arrangement)
    return population

def evaluate_fitness(arrangement, matches):
    """Calculate fitness score for an arrangement."""
    fitness = 0
    for i in range(len(arrangement) - 1):
        piece1, piece2 = arrangement[i], arrangement[i + 1]
        match = matches.get((piece1['id'], 'right'))
        if match and match[0] == piece2['id']:
            fitness -= 1  # Better match reduces the fitness score
    return fitness

def genetic_algorithm(pieces, matches, generations=100, population_size=10):
    """Optimize arrangement using Genetic Algorithm."""
    population = initialize_population(pieces, population_size)
    for _ in range(generations):
        # Evaluate fitness
        fitness_scores = [(arrangement, evaluate_fitness(arrangement, matches)) for arrangement in population]
        fitness_scores.sort(key=lambda x: x[1])

        # Selection
        selected = fitness_scores[:len(fitness_scores) // 2]

        # Crossover and Mutation
        next_generation = []
        for i in range(len(selected) // 2):
            parent1 = selected[2 * i][0]
            parent2 = selected[2 * i + 1][0]
            midpoint = len(parent1) // 2
            child = parent1[:midpoint] + parent2[midpoint:]
            if random.random() < 0.1:
                swap_idx1, swap_idx2 = random.sample(range(len(child)), 2)
                child[swap_idx1], child[swap_idx2] = child[swap_idx2], child[swap_idx1]
            next_generation.append(child)
        population = [x[0] for x in selected] + next_generation

    # Return best arrangement
    return sorted(population, key=lambda arr: evaluate_fitness(arr, matches))[0]

def assemble_puzzle(best_arrangement):
    """Assemble the puzzle based on the best arrangement."""
    # Determine the size of the final image
    piece_height, piece_width = best_arrangement[0]['image'].shape[:2]
    rows = int(np.sqrt(len(best_arrangement)))
    cols = rows  # Assume a square puzzle for simplicity
    puzzle_height = rows * piece_height
    puzzle_width = cols * piece_width

    # Create a blank canvas for the final image
    final_image = np.zeros((puzzle_height, puzzle_width, 3), dtype=np.uint8)

    # Place each piece in its corresponding location
    for idx, piece in enumerate(best_arrangement):
        row = idx // cols
        col = idx % cols
        x_start = col * piece_width
        y_start = row * piece_height
        x_end = x_start + piece_width
        y_end = y_start + piece_height
        final_image[y_start:y_end, x_start:x_end] = piece['image']

    return final_image

if __name__ == "__main__":
    # Load and process puzzle pieces
    piece_paths = os.listdir('pieces')  # Example paths
    all_edges = []

    for i, path in enumerate(piece_paths):
        piece, contour = remove_white_background(path)
        edges = extract_edges_from_contour(contour)
        all_edges.append({"id": i, "edges": edges, "image": piece})

    # Find best matches
    matches = find_best_matches(all_edges)

    # Solve the puzzle using Genetic Algorithm
    best_arrangement = genetic_algorithm(all_edges, matches)

    # Assemble the puzzle image
    final_image = assemble_puzzle(best_arrangement)

    # Display the final image
    cv2.imshow("Assembled Puzzle", final_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Save the final image
    cv2.imwrite("assembled_puzzle.png", final_image)
    print("Assembled puzzle saved as 'assembled_puzzle.png'")


