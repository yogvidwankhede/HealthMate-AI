# ============================================================================
# HealthMate-AI: Evaluation with Data Persistence
# Run evaluation ONCE, then just load and visualize
# ============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
import os
import matplotlib.patches as patches

# ============================================================================
# GLOBAL VARIABLE TO CACHE RESULTS
# ============================================================================

_cached_results_df = None  # Store results in memory for fast updates


def calculate_bleu_simple(reference, generated):
    """Simple BLEU Score: Measures word overlap (0-1)"""
    ref_words = set(reference.lower().split())
    gen_words = set(generated.lower().split())

    if len(gen_words) == 0:
        return 0

    matches = len(ref_words & gen_words)
    bleu = matches / len(gen_words)

    return bleu


# ============================================================================
# METRIC 2: ROUGE-L F1 SCORE
# ============================================================================

def calculate_rouge_l_simple(reference, generated):
    """ROUGE-L F1: Measures sentence structure match (0-1)"""
    ref_tokens = reference.lower().split()
    gen_tokens = generated.lower().split()

    if len(gen_tokens) == 0 or len(ref_tokens) == 0:
        return 0

    def lcs_length(a, b):
        m, n = len(a), len(b)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if a[i-1] == b[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        return dp[m][n]

    lcs = lcs_length(ref_tokens, gen_tokens)

    precision = lcs / len(gen_tokens) if len(gen_tokens) > 0 else 0
    recall = lcs / len(ref_tokens) if len(ref_tokens) > 0 else 0

    f1 = 2 * (precision * recall) / (precision +
                                     recall) if (precision + recall) > 0 else 0

    return f1


# ============================================================================
# SAVE EVALUATION RESULTS
# ============================================================================

def save_evaluation_results(results_df, results_folder='evaluation_results'):
    """
    Save evaluation results to persistent storage (CSV + JSON)
    
    This saves once, so you don't need to run evaluation again!
    """

    # Create folder if it doesn't exist
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 1. Save full results as CSV
    csv_file = f'{results_folder}/evaluation_results_{timestamp}.csv'
    results_df.to_csv(csv_file, index=False)
    print(f"✅ Saved CSV: {csv_file}")

    # 2. Save summary statistics as JSON
    summary = {
        'metadata': {
            'evaluation_date': datetime.now().isoformat(),
            'total_queries': len(results_df),
            'results_file': csv_file
        },
        'metrics': {
            'BLEU': {
                'average': float(results_df['BLEU_Score'].mean()),
                'min': float(results_df['BLEU_Score'].min()),
                'max': float(results_df['BLEU_Score'].max()),
                'std': float(results_df['BLEU_Score'].std())
            },
            'ROUGE_L': {
                'average': float(results_df['ROUGE_L_Score'].mean()),
                'min': float(results_df['ROUGE_L_Score'].min()),
                'max': float(results_df['ROUGE_L_Score'].max()),
                'std': float(results_df['ROUGE_L_Score'].std())
            },
            'Max_Score': {
                'average': float(results_df['Max_Score'].mean()),
                'min': float(results_df['Max_Score'].min()),
                'max': float(results_df['Max_Score'].max()),
                'std': float(results_df['Max_Score'].std())
            }
        },
        'passing_criteria': {
            'BLEU_target_0.5': bool(results_df['BLEU_Score'].mean() >= 0.5),
            'ROUGE_L_target_0.5': bool(results_df['ROUGE_L_Score'].mean() >= 0.5)
        }
    }

    json_file = f'{results_folder}/evaluation_summary_{timestamp}.json'
    with open(json_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"✅ Saved JSON: {json_file}")

    # 3. Create a latest results symlink/copy
    latest_csv = f'{results_folder}/evaluation_results_LATEST.csv'
    latest_json = f'{results_folder}/evaluation_summary_LATEST.json'

    results_df.to_csv(latest_csv, index=False)
    with open(latest_json, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"✅ Saved latest files:")
    print(f"   • {latest_csv}")
    print(f"   • {latest_json}")

    return csv_file, json_file


# ============================================================================
# VISUALIZATION 2: BOX AND WHISKERS PLOT
# ============================================================================

def visualize_box_whiskers(results_folder='evaluation_results'):
    """
    Load saved results and create box and whiskers plot
    Using MAX_SCORE instead of average
    Shows median, quartiles, and outliers for each metric
    """

    print("📊 Creating Box and Whiskers Plot from saved data...\n")
    results_df = load_evaluation_results(results_folder, use_latest=True)

    if results_df is None:
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    fig.suptitle('HealthMate-AI: Box and Whiskers Analysis',
                 fontsize=16, fontweight='bold')

    # Prepare data
    bleu_data = results_df['BLEU_Score'].dropna()
    rouge_data = results_df['ROUGE_L_Score'].dropna()
    max_data = results_df['Max_Score'].dropna()

    # ============================================================
    # 1. BLEU Score Box Plot
    # ============================================================
    ax1 = axes[0]

    bp1 = ax1.boxplot(bleu_data,
                      vert=True,
                      patch_artist=True,
                      widths=0.5,
                      labels=['BLEU Score'],
                      medianprops=dict(color='red', linewidth=2),
                      boxprops=dict(facecolor='#3498db', alpha=0.7,
                                    edgecolor='black', linewidth=2),
                      whiskerprops=dict(color='black', linewidth=1.5),
                      capprops=dict(color='black', linewidth=1.5),
                      flierprops=dict(marker='o', markerfacecolor='red', markersize=6, linestyle='none', markeredgecolor='red'))

    ax1.set_ylabel('Score Value', fontsize=11, fontweight='bold')
    ax1.set_title('BLEU Score Distribution', fontsize=12, fontweight='bold')
    ax1.set_ylim([0, 1])
    ax1.grid(axis='y', alpha=0.3, linestyle='--')

    # Add statistics text
    stats_text1 = f"""
    Min: {bleu_data.min():.3f}
    Q1: {bleu_data.quantile(0.25):.3f}
    Median: {bleu_data.median():.3f}
    Q3: {bleu_data.quantile(0.75):.3f}
    Max: {bleu_data.max():.3f}
    IQR: {bleu_data.quantile(0.75) - bleu_data.quantile(0.25):.3f}
    """
    ax1.text(1.3, 0.5, stats_text1, fontsize=9, family='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # ============================================================
    # 2. ROUGE-L Score Box Plot
    # ============================================================
    ax2 = axes[1]

    bp2 = ax2.boxplot(rouge_data,
                      vert=True,
                      patch_artist=True,
                      widths=0.5,
                      labels=['ROUGE-L Score'],
                      medianprops=dict(color='red', linewidth=2),
                      boxprops=dict(facecolor='#9b59b6', alpha=0.7,
                                    edgecolor='black', linewidth=2),
                      whiskerprops=dict(color='black', linewidth=1.5),
                      capprops=dict(color='black', linewidth=1.5),
                      flierprops=dict(marker='o', markerfacecolor='red', markersize=6, linestyle='none', markeredgecolor='red'))

    ax2.set_ylabel('Score Value', fontsize=11, fontweight='bold')
    ax2.set_title('ROUGE-L Score Distribution', fontsize=12, fontweight='bold')
    ax2.set_ylim([0, 1])
    ax2.grid(axis='y', alpha=0.3, linestyle='--')

    # Add statistics text
    stats_text2 = f"""
    Min: {rouge_data.min():.3f}
    Q1: {rouge_data.quantile(0.25):.3f}
    Median: {rouge_data.median():.3f}
    Q3: {rouge_data.quantile(0.75):.3f}
    Max: {rouge_data.max():.3f}
    IQR: {rouge_data.quantile(0.75) - rouge_data.quantile(0.25):.3f}
    """
    ax2.text(1.3, 0.5, stats_text2, fontsize=9, family='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # ============================================================
    # 3. Max Score Box Plot
    # ============================================================
    ax3 = axes[2]

    bp3 = ax3.boxplot(max_data,
                      vert=True,
                      patch_artist=True,
                      widths=0.5,
                      labels=['Max Score'],
                      medianprops=dict(color='red', linewidth=2),
                      boxprops=dict(facecolor='#2ecc71', alpha=0.7,
                                    edgecolor='black', linewidth=2),
                      whiskerprops=dict(color='black', linewidth=1.5),
                      capprops=dict(color='black', linewidth=1.5),
                      flierprops=dict(marker='o', markerfacecolor='red', markersize=6, linestyle='none', markeredgecolor='red'))

    ax3.set_ylabel('Score Value', fontsize=11, fontweight='bold')
    ax3.set_title('Max Score Distribution', fontsize=12, fontweight='bold')
    ax3.set_ylim([0, 1])
    ax3.grid(axis='y', alpha=0.3, linestyle='--')

    # Add statistics text
    stats_text3 = f"""
    Min: {max_data.min():.3f}
    Q1: {max_data.quantile(0.25):.3f}
    Median: {max_data.median():.3f}
    Q3: {max_data.quantile(0.75):.3f}
    Max: {max_data.max():.3f}
    IQR: {max_data.quantile(0.75) - max_data.quantile(0.25):.3f}
    """
    ax3.text(1.3, 0.5, stats_text3, fontsize=9, family='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig('evaluation_box_whiskers.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: evaluation_box_whiskers.png\n")
    plt.show()


# ============================================================================
# BOX PLOT INTERPRETATION GUIDE
# ============================================================================

def print_box_whiskers_guide():
    """
    Print guide on how to interpret box and whiskers plots
    """

    guide = """
╔════════════════════════════════════════════════════════════════════╗
║           BOX AND WHISKERS PLOT - INTERPRETATION GUIDE             ║
╚════════════════════════════════════════════════════════════════════╝

PARTS OF A BOX PLOT:
────────────────────

                    ◆ (Outlier)
                    |
        Whisker ────┤
                    |
    ┌───────────────┤ ← Upper Quartile (Q3 - 75%)
    │               │
    │   ┌─────────┐ │ ← Interquartile Range (IQR)
    │   │ Box     │ │    Contains 50% of data
    │   │─────────│ │ ← Median (Q2 - 50%)
    │   │         │ │
    └───┤─────────┤─┤ ← Lower Quartile (Q1 - 25%)
        │         │ │
        Whisker ──┘─┤
                    |
                    ◆ (Outlier)


KEY STATISTICS:
───────────────

Min Value:    Smallest data point (excluding outliers)
Q1 (25%):     Lower quartile - 25% of data below this
Median:       Middle value - 50% of data on each side (RED LINE)
Q3 (75%):     Upper quartile - 75% of data below this
Max Value:    Largest data point (excluding outliers)
IQR:          Interquartile Range = Q3 - Q1 (width of box)
Outliers:     Red dots above/below whiskers


WHAT IT SHOWS:
──────────────

✓ Central Tendency    → Median line position
✓ Spread/Variability  → Box width (IQR)
✓ Symmetry           → Position of median line in box
✓ Outliers           → Individual red dots
✓ Data Range         → Distance between whiskers


INTERPRETATION EXAMPLES:
────────────────────────

EXAMPLE 1: Consistent Performance
───────────────────────────────────
    ┌─────────┐
    │────●────│ ← Median centered, box compact
    └─────────┘
    → Good, consistent performance
    → Most answers have similar quality


EXAMPLE 2: High Variability
─────────────────────────────
    ┌──────────────────┐
    │─●────────────────│ ← Median off-center, large box
    └──────────────────┘
    → High variance
    → Mix of very good and very poor answers


EXAMPLE 3: Skewed Right (Right-Skewed)
────────────────────────────────────────
    ┌────────────┐
    │──●─────────│ ← Median toward left
    └────────────┘
    → Many good scores, few poor ones
    → System performs well


EXAMPLE 4: Skewed Left (Left-Skewed)
──────────────────────────────────────
    ┌────────────┐
    │─────●──────│ ← Median toward right
    └────────────┘
    → Many poor scores, few good ones
    → System needs improvement


OUTLIERS (Red Dots):
────────────────────

Outliers are calculated as:
  Below: Q1 - 1.5 × IQR
  Above: Q3 + 1.5 × IQR

Single outliers:   → Worth investigating
Many outliers:     → Data is highly variable


COMPARING MULTIPLE BOX PLOTS:
──────────────────────────────

BLEU vs ROUGE-L vs Average:

If BLEU box is HIGHER than ROUGE-L:
  → BLEU scores better (more word overlap)
  → ROUGE-L is stricter (cares about order)

If Average box is in MIDDLE:
  → Good - represents both metrics

If one metric has MORE outliers:
  → That metric is less consistent


WHAT IT MEANS FOR YOUR SYSTEM:
──────────────────────────────

GOOD SIGNS:
  ✓ Median > 0.5 (mostly passing)
  ✓ Small box (consistent performance)
  ✓ Few outliers (predictable)
  ✓ Box in upper half (good quality)

BAD SIGNS:
  ✗ Median < 0.5 (mostly failing)
  ✗ Large box (inconsistent performance)
  ✗ Many outliers (unpredictable)
  ✗ Box in lower half (poor quality)

NEEDS IMPROVEMENT:
  ⚠ Median around 0.5 (borderline)
  ⚠ Box touching 0 or 1 (extreme range)
  ⚠ Left-skewed (too many poor answers)
  ⚠ Outliers below 0.3 (dangerous failures)


╔════════════════════════════════════════════════════════════════════╗
║ For your HealthMate-AI system, the box plot shows reliability and ║
║ consistency - critical for medical AI safety!                     ║
╚════════════════════════════════════════════════════════════════════╝
"""

    print(guide)


def load_evaluation_results(results_folder='evaluation_results', use_latest=True):
    """
    Load previously saved evaluation results
    
    Args:
        results_folder: Folder where results are stored
        use_latest: If True, loads latest results; else loads all
    
    Returns:
        DataFrame with evaluation results
    """

    if not os.path.exists(results_folder):
        print(f"❌ Results folder not found: {results_folder}")
        print("💡 Run evaluation first: run_full_evaluation()")
        return None

    if use_latest:
        latest_csv = f'{results_folder}/evaluation_results_LATEST.csv'
        if os.path.exists(latest_csv):
            print(f"📂 Loading latest results: {latest_csv}\n")
            results_df = pd.read_csv(latest_csv)
            return results_df
        else:
            print(f"⚠️  Latest file not found. Looking for other files...\n")

    # Find all CSV files
    csv_files = [f for f in os.listdir(results_folder) if f.startswith(
        'evaluation_results_') and f.endswith('.csv')]

    if not csv_files:
        print(f"❌ No evaluation results found in: {results_folder}")
        return None

    # Load most recent file
    latest_file = sorted(csv_files)[-1]
    file_path = f'{results_folder}/{latest_file}'

    print(f"📂 Loading: {file_path}\n")
    results_df = pd.read_csv(file_path)

    return results_df


# ============================================================================
# QUICK VISUALIZATION FUNCTIONS (Load cached data only)
# ============================================================================

def quick_visualize_all(results_folder='evaluation_results'):
    """
    Load cached results and show ALL visualizations instantly
    
    Perfect for tweaking visualization code!
    No re-evaluation needed.
    
    Usage:
        quick_visualize_all()
    """

    print("⚡ Quick loading visualizations (no evaluation)...\n")

    results_df = load_evaluation_results(results_folder, use_latest=True)

    if results_df is None:
        return

    print("📊 Showing all visualizations...\n")

    # Show report
    generate_report_from_saved(results_folder)

    # Show all visualizations
    visualize_metrics_from_saved(results_folder)
    visualize_box_whiskers(results_folder)

    print("\n✅ All visualizations loaded!")
    print("💡 Make changes to visualization code and run quick_visualize_all() again\n")


def quick_reload():
    """
    Reload cached data into memory
    
    Usage:
        quick_reload()
        visualize_metrics_from_saved()
    """

    global _cached_results_df
    _cached_results_df = None  # Clear cache

    print("🔄 Reloading data...\n")
    load_evaluation_results()
    print("✅ Data reloaded into memory\n")


def main_evaluation(csv_path='new_medical_questions.csv', rag_chain=None, retriever=None,
                    encoding='macroman', results_folder='evaluation_results', force_rerun=False):
    """
    Smart evaluation function - runs ONCE, then loads cached results
    
    Args:
        csv_path: Path to CSV
        rag_chain: RAG chain (not needed if results exist)
        retriever: Retriever (not needed if results exist)
        encoding: CSV encoding
        results_folder: Where to save/load results
        force_rerun: If True, delete old results and re-evaluate
    
    Returns:
        DataFrame with results
    """

    latest_results_file = f'{results_folder}/evaluation_results_LATEST.csv'

    # Check if results already exist
    if os.path.exists(latest_results_file) and not force_rerun:
        print("\n" + "="*70)
        print("✅ EVALUATION RESULTS FOUND!")
        print("="*70)
        print(f"📂 Loading from: {latest_results_file}\n")

        results_df = pd.read_csv(latest_results_file)

        print(f"✓ Loaded {len(results_df)} evaluation results")
        print(f"📊 Displaying report and visualizations...\n")

        # Show report
        generate_report_from_saved(results_folder)

        # Show visualizations
        print("📊 Creating visualizations...\n")
        visualize_metrics_from_saved(results_folder)
        visualize_box_whiskers(results_folder)

        return results_df

    # If results don't exist or force_rerun is True, run evaluation
    if force_rerun:
        print(f"\n⚠️  Force re-run requested. Deleting old results...\n")
        if os.path.exists(latest_results_file):
            os.remove(latest_results_file)
        print("Deleting JSON files...\n")
        json_file = f'{results_folder}/evaluation_summary_LATEST.json'
        if os.path.exists(json_file):
            os.remove(json_file)

    # Check if required arguments are provided
    if rag_chain is None or retriever is None:
        print("\n❌ ERROR: First run requires rag_chain and retriever arguments!")
        print("\nUsage on first run:")
        print("  results_df = main_evaluation(")
        print("      rag_chain=rag_chain,")
        print("      retriever=retriever,")
        print("      encoding='macroman'")
        print("  )")
        return None

    print("\n" + "="*70)
    print("🏥 HealthMate-AI - FULL EVALUATION (First Run)")
    print("="*70 + "\n")

    return run_full_evaluation(csv_path, rag_chain, retriever, encoding, results_folder)


def run_full_evaluation(csv_path, rag_chain, retriever, encoding='macroman', results_folder='evaluation_results'):
    """
    Run complete evaluation and save results
    
    ONLY RUN THIS ONCE!
    
    Args:
        csv_path: Path to new_medical_questions.csv
        rag_chain: Your RAG chain from trials.ipynb
        retriever: Your Pinecone retriever from trials.ipynb
        encoding: CSV encoding (default: 'macroman')
        results_folder: Where to save results
    """

    print("🏥 HealthMate-AI - FULL EVALUATION")
    print("="*70)
    print("⚠️  This runs only ONCE and saves all data\n")

    # Load CSV
    print("📂 Loading CSV data...")
    print(f"📋 Using encoding: {encoding}\n")

    df = pd.read_csv(csv_path, encoding=encoding)
    df.columns = df.columns.str.strip()

    print(f"✓ Loaded {len(df)} questions from CSV")
    print(f"✓ Columns: {list(df.columns)}\n")

    # Initialize results
    results = []

    # Evaluate each query
    print("🔄 Evaluating questions...\n")

    for idx, row in df.iterrows():
        query = row['query'].strip()
        csv_answer = row['reference_answer'].strip()
        category = row['category'].strip()
        difficulty = row['difficulty'].strip()

        try:
            # Get retrieval results and RAG answer
            retrieved_docs = retriever.invoke(query)
            retrieved_text = " ".join(
                [doc.page_content for doc in retrieved_docs])

            rag_response = rag_chain.invoke({"input": query})
            rag_answer = rag_response["answer"]

            # Calculate metrics
            bleu_score = calculate_bleu_simple(csv_answer, rag_answer)
            rouge_score = calculate_rouge_l_simple(csv_answer, rag_answer)

            results.append({
                'ID': row['id'],
                'Query': query,
                'Category': category,
                'Difficulty': difficulty,
                'CSV_Answer': csv_answer,
                'RAG_Answer': rag_answer,
                'BLEU_Score': bleu_score,
                'ROUGE_L_Score': rouge_score,
                'Max_Score': max(bleu_score, rouge_score)
            })

            print(f"✓ [{idx+1}/{len(df)}] {query[:60]}...")
            print(f"    BLEU: {bleu_score:.3f} | ROUGE-L: {rouge_score:.3f}\n")

        except Exception as e:
            print(f"✗ [{idx+1}/{len(df)}] Error: {str(e)}\n")
            results.append({
                'ID': row['id'],
                'Query': query,
                'Category': category,
                'Difficulty': difficulty,
                'CSV_Answer': csv_answer,
                'RAG_Answer': 'ERROR',
                'BLEU_Score': 0,
                'ROUGE_L_Score': 0,
                'Max_Score': 0
            })

    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    # Save results
    print("\n" + "="*70)
    print("💾 SAVING EVALUATION RESULTS")
    print("="*70 + "\n")

    save_evaluation_results(results_df, results_folder)

    print("\n✅ EVALUATION COMPLETE!")
    print("⚡ Next time just load and visualize - no need to re-evaluate!\n")

    return results_df


# ============================================================================
# GENERATE REPORT FROM SAVED DATA
# ============================================================================

def generate_report_from_saved(results_folder='evaluation_results'):
    """
    Load saved results and print report
    
    NO EVALUATION - INSTANT!
    """

    results_df = load_evaluation_results(results_folder, use_latest=True)

    if results_df is None:
        return

    print("\n" + "="*70)
    print("HEALTHMATE-AI EVALUATION REPORT (from saved data)")
    print("="*70)
    print(f"\nTotal Questions: {len(results_df)}")

    print("\n" + "-"*70)
    print("METRIC SUMMARY")
    print("-"*70)

    print(f"\nBLEU Score:")
    print(f"  Average:  {results_df['BLEU_Score'].mean():.4f}")
    print(f"  Min:      {results_df['BLEU_Score'].min():.4f}")
    print(f"  Max:      {results_df['BLEU_Score'].max():.4f}")

    print(f"\nROUGE-L Score:")
    print(f"  Average:  {results_df['ROUGE_L_Score'].mean():.4f}")
    print(f"  Min:      {results_df['ROUGE_L_Score'].min():.4f}")
    print(f"  Max:      {results_df['ROUGE_L_Score'].max():.4f}")

    print(f"\nOverall Maximum Score: {results_df['Max_Score'].mean():.4f}")

    print("\n" + "-"*70)
    print("PASSING CRITERIA")
    print("-"*70)
    bleu_pass = (results_df['BLEU_Score'].mean() >= 0.5)
    rouge_pass = (results_df['ROUGE_L_Score'].mean() >= 0.5)

    print(f"BLEU ≥ 0.50:    {'✓ PASS' if bleu_pass else '✗ FAIL'}")
    print(f"ROUGE-L ≥ 0.50: {'✓ PASS' if rouge_pass else '✗ FAIL'}")

    print("\n" + "-"*70)
    print("TOP 5 BEST PERFORMING")
    print("-"*70)

    top_5 = results_df.nlargest(5, 'Max_Score')[
        ['Query', 'BLEU_Score', 'ROUGE_L_Score', 'Max_Score']]
    for idx, (i, row) in enumerate(top_5.iterrows(), 1):
        print(f"\n{idx}. {row['Query'][:60]}...")
        print(
            f"   BLEU: {row['BLEU_Score']:.3f} | ROUGE-L: {row['ROUGE_L_Score']:.3f} | Max: {row['Max_Score']:.3f}")

    print("\n" + "="*70 + "\n")

    return results_df


# ============================================================================
# VISUALIZATIONS (Load from saved data)
# ============================================================================

def visualize_metrics_from_saved(results_folder='evaluation_results'):
    """Load saved results and create visualizations"""

    print("📊 Loading saved results for visualization...\n")
    results_df = load_evaluation_results(results_folder, use_latest=True)

    if results_df is None:
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('HealthMate-AI: Evaluation Metrics Analysis',
                 fontsize=16, fontweight='bold')

    # 1. Average metrics
    ax1 = axes[0, 0]
    metrics = {
        'BLEU': results_df['BLEU_Score'].mean(),
        'ROUGE-L': results_df['ROUGE_L_Score'].mean()
    }
    colors = ['#2ecc71' if v >= 0.5 else '#e74c3c' for v in metrics.values()]
    bars = ax1.bar(metrics.keys(), metrics.values(), color=colors,
                   alpha=0.7, edgecolor='black', linewidth=2)
    ax1.axhline(y=0.5, color='red', linestyle='--',
                linewidth=2, label='Target: 0.50')
    ax1.set_ylabel('Score', fontsize=11, fontweight='bold')
    ax1.set_title('Average Metric Scores', fontsize=12, fontweight='bold')
    ax1.set_ylim([0, 1])
    ax1.legend()

    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.3f}', ha='center', va='bottom', fontweight='bold')

    # 2. BLEU distribution
    ax2 = axes[0, 1]
    ax2.hist(results_df['BLEU_Score'], bins=15,
             color='#3498db', alpha=0.7, edgecolor='black')
    ax2.axvline(results_df['BLEU_Score'].mean(), color='red', linestyle='--', linewidth=2,
                label=f'Mean: {results_df["BLEU_Score"].mean():.3f}')
    ax2.set_xlabel('BLEU Score', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax2.set_title('BLEU Score Distribution', fontsize=12, fontweight='bold')
    ax2.legend()

    # 3. ROUGE-L distribution
    ax3 = axes[1, 0]
    ax3.hist(results_df['ROUGE_L_Score'], bins=15,
             color='#9b59b6', alpha=0.7, edgecolor='black')
    ax3.axvline(results_df['ROUGE_L_Score'].mean(), color='red', linestyle='--', linewidth=2,
                label=f'Mean: {results_df["ROUGE_L_Score"].mean():.3f}')
    ax3.set_xlabel('ROUGE-L Score', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax3.set_title('ROUGE-L Score Distribution', fontsize=12, fontweight='bold')
    ax3.legend()

    # 4. Scatter plot
    ax4 = axes[1, 1]
    scatter = ax4.scatter(results_df['BLEU_Score'], results_df['ROUGE_L_Score'],
                          s=100, alpha=0.6, c=results_df['Max_Score'],
                          cmap='RdYlGn', edgecolors='black', linewidth=1)
    ax4.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Perfect Match')

    # Add danger zone rectangle
    square = patches.Rectangle(
        (0, 0), 0.15, 0.15, linewidth=2, edgecolor='r', facecolor='none', linestyle='--')
    ax4.add_patch(square)
    ax4.text(0.075, -0.05, 'Danger Zone', ha='center',
             fontsize=9, color='red', fontweight='bold')

    ax4.set_xlabel('BLEU Score', fontsize=11, fontweight='bold')
    ax4.set_ylabel('ROUGE-L Score', fontsize=11, fontweight='bold')
    ax4.set_title('BLEU vs ROUGE-L', fontsize=12, fontweight='bold')
    ax4.set_xlim([0, 1])
    ax4.set_ylim([0, 1])
    cbar = plt.colorbar(scatter, ax=ax4)
    cbar.set_label('Max Score', fontweight='bold')
    ax4.legend()

    plt.tight_layout()
    plt.savefig('evaluation_metrics.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: evaluation_metrics.png\n")
    plt.show()


# ============================================================================
# USAGE GUIDE
# ============================================================================

"""
WORKFLOW:

STEP 1: FIRST TIME - Smart function runs evaluation automatically
──────────────────────────────────────────────────────────────────
results_df = main_evaluation(
    csv_path='new_medical_questions.csv',
    rag_chain=rag_chain,
    retriever=retriever,
    encoding='macroman'
)

→ Runs evaluation, saves to evaluation_results/
→ Shows report and visualizations


STEP 2: NEXT TIME - Just call the same function (loads instantly)
──────────────────────────────────────────────────────────────────
results_df = main_evaluation()

→ Automatically detects saved data
→ Loads results instantly (< 1 second)
→ Shows report and visualizations


THAT'S IT! No code changes needed!


ADVANCED OPTIONS:
─────────────────

Force re-evaluation (delete old results and run again):
results_df = main_evaluation(force_rerun=True)

Load results without running evaluation:
results_df = load_evaluation_results()

Just show visualizations:
visualize_metrics_from_saved()
visualize_box_whiskers()
"""
