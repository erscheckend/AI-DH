import csv
import sys
from sacrebleu import sentence_bleu
from rouge_score import rouge_scorer
from sacrebleu.metrics import CHRF
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize
import nltk

# Define individual score calculation functions
def calculate_bleu(reference, hypothesis):
    return sentence_bleu(hypothesis, [reference]).score

def calculate_rouge_l(reference, hypothesis):
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = scorer.score(reference, hypothesis)
    return scores['rougeL'].fmeasure * 100

def calculate_chrf(reference, hypothesis):
    chrf_scorer = CHRF()
    score = chrf_scorer.sentence_score(hypothesis, [reference])
    return score.score

def calculate_meteor(reference, hypothesis):
    reference_tokens = word_tokenize(reference)
    hypothesis_tokens = word_tokenize(hypothesis)
    return meteor_score([reference_tokens], hypothesis_tokens) * 100

# Unified function to process the CSV, calculate all metrics, and save results
def process_csv(file_path):
    output_rows = []  # List to collect output rows

    with open(file_path, 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile, delimiter=';')
        headers = next(reader)
        
        # Append new headers for the metrics to the existing headers
        output_rows.append(headers + [
            "BLEU score GPT", "BLEU score Google",
            "ROUGE score GPT", "ROUGE score Google",
            "chrF scoreGPT", "chrF score Google",
            "METEOR scoreGPT", "METEOR score Google"
        ])
        
        for row_num, row in enumerate(reader, start=2):
            # Ensure row has exactly six columns
            if len(row) != 6:
                print(f"Skipping row {row_num} due to unexpected number of columns ({len(row)} columns found). Row content: {row}")
                continue
            
            # Extract reference and hypotheses from specific columns
            reference = row[3]
            hypothesis_gpt = row[4]
            hypothesis_google = row[5]

            # Calculate each metric
            bleu_gpt = round(calculate_bleu(reference, hypothesis_gpt), 2)
            bleu_google = round(calculate_bleu(reference, hypothesis_google), 2)

            rouge_gpt = round(calculate_rouge_l(reference, hypothesis_gpt), 2)
            rouge_google = round(calculate_rouge_l(reference, hypothesis_google), 2)

            chrf_gpt = round(calculate_chrf(reference, hypothesis_gpt), 2)
            chrf_google = round(calculate_chrf(reference, hypothesis_google), 2)

            meteor_gpt = round(calculate_meteor(reference, hypothesis_gpt), 2)
            meteor_google = round(calculate_meteor(reference, hypothesis_google), 2)

            # Append calculated scores to the row
            output_rows.append(row + [
                bleu_gpt, bleu_google,
                rouge_gpt, rouge_google,
                chrf_gpt, chrf_google,
                meteor_gpt, meteor_google
            ])

    # Write the output to a new CSV file with "_with_scores" suffix
    output_file_path = file_path.replace('.csv', '_with_scores.csv')
    with open(output_file_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile, delimiter=';')
        writer.writerows(output_rows)

    print(f"Evaluation results saved to {output_file_path}")

# Main entry point
def main():
    if len(sys.argv) != 2:
        print("Usage: python script.py <csv_file_path>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    process_csv(file_path)

if __name__ == "__main__":
    main()
