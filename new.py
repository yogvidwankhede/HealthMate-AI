# In trials.ipynb, add this at the end:
from metrics import evaluate_rag_system, generate_summary_report, visualize_metrics_distribution, visualize_performance_categories, save_results

results_df = evaluate_rag_system(
    csv_path='data/new_medical_questions.csv',
    rag_chain=rag_chain,
    retriever=retriever,
)

generate_summary_report(results_df)
visualize_metrics_distribution(results_df)
visualize_performance_categories(results_df)
# visualize_by_category_and_difficulty(results_df)
save_results(results_df)
