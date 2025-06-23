import xml.etree.ElementTree as ET
from pathlib import Path
import torch
import re
import pandas as pd
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer
from datasets import Dataset
from evaluate import load
import unicodedata
from difflib import SequenceMatcher


# --- Dataset Preparation ---
def create_training_data(input_dir, ref_dir, output_file="data/train.en-de.tsv"):
    """Extract parallel English-German texts from XML files."""
    input_dir = Path(input_dir)
    ref_dir = Path(ref_dir)
    output_file = Path(output_file)
    output_file.parent.mkdir(exist_ok=True)

    data = {"en_text": [], "de_text": []}
    translate_tags = {'title', 'head', 'p', 'note'}
    processed_files = []

    def clean_text(text):
        """Clean text by removing special characters and normalizing."""
        if not text:
            return ""
        text = unicodedata.normalize("NFKC", text)
        text = re.sub(r'[\x00-\x1F\x7F]', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    for en_file in input_dir.glob("*-en.xml"):
        de_file = ref_dir / en_file.name.replace("-en.xml", "-de.xml")
        if not de_file.exists():
            print(f"Missing reference: {de_file}")
            continue

        try:
            en_tree = ET.parse(en_file)
            de_tree = ET.parse(de_file)

            en_texts = []
            de_texts = []
            for en_elem, de_elem in zip(en_tree.iter(), de_tree.iter()):
                if en_elem.tag in translate_tags and en_elem.text and de_elem.text:
                    en_text = clean_text(en_elem.text)
                    de_text = clean_text(de_elem.text)
                    if en_text and de_text and len(en_text) < 500 and len(de_text) < 500:
                        en_texts.append(en_text)
                        de_texts.append(de_text)

            if en_texts:
                data["en_text"].extend(en_texts)
                data["de_text"].extend(de_texts)
                processed_files.append(en_file.name)
        except Exception as e:
            print(f"Error parsing {en_file}: {str(e)}")
            continue

    legal_terms = {
        "Official Journal": "Amtsblatt",
        "paper edition": "Papierausgabe",
        "authentic": "verbindlich",
        "deemed authentic": "verbindlich gelten",
        "regulation": "Verordnung"
    }
    data["en_text"].extend(list(legal_terms.keys()))
    data["de_text"].extend(list(legal_terms.values()))

    df = pd.DataFrame(data)
    if not df.empty:
        df = df.drop_duplicates().dropna()
        df.to_csv(output_file, sep="\t", index=False)
        print(f"Saved {len(df)} sentence pairs to {output_file}")
        print(f"Processed files: {', '.join(processed_files)}")
    else:
        print("No valid data extracted. Using legal terms only.")
        df = pd.DataFrame({
            "en_text": list(legal_terms.keys()),
            "de_text": list(legal_terms.values())
        })
        df.to_csv(output_file, sep="\t", index=False)
        print(f"Saved {len(df)} legal terms to {output_file}")
    return df


# --- Fine-Tuning ---
def fine_tune_model(data_path="data/train.en-de.tsv", output_dir="finetuned_model"):
    """Fine-tune the translation model."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    try:
        df = pd.read_csv(data_path, sep="\t")
        dataset = {"translation": [{"en": row["en_text"], "de": row["de_text"]} for _, row in df.iterrows() if
                                   row["en_text"] and row["de_text"]]}
        train_dataset = Dataset.from_dict(dataset)
        print(f"Loaded {len(train_dataset)} sentence pairs for training")
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        return

    model_name = "Helsinki-NLP/opus-mt-en-de"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # 🔒 Freeze the encoder layers
    if hasattr(model, "model") and hasattr(model.model, "encoder"):
        for param in model.model.encoder.parameters():
            param.requires_grad = False
        print("Encoder has been frozen.")
    else:
        print("Warning: Model does not have expected encoder structure. No layers frozen.")

    # ✅ Optional: Check what's still trainable
    trainable_params = [name for name, param in model.named_parameters() if param.requires_grad]
    print(f"Number of trainable parameters: {len(trainable_params)}")
    for name in trainable_params:
        print(name)

    # 🔁 Add the rest of your training logic (tokenization, training args, trainer, trainer.train(), etc.)


    def preprocess_function(examples):
        inputs = [ex["en"] for ex in examples["translation"]]
        targets = [ex["de"] for ex in examples["translation"]]
        model_inputs = tokenizer(inputs, max_length=256, truncation=True, padding="max_length")

        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, max_length=256, truncation=True, padding="max_length")["input_ids"]
            labels = [[(label if label != tokenizer.pad_token_id else -100) for label in seq] for seq in labels]

        model_inputs["labels"] = labels
        return model_inputs

    try:
        tokenized_train = train_dataset.map(preprocess_function, batched=True, remove_columns=["translation"])
    except Exception as e:
        print(f"Tokenization failed: {str(e)}")
        return

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        eval_strategy="no",
        learning_rate=5e-6,
        per_device_train_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        save_total_limit=1,
        predict_with_generate=True,
        fp16=torch.cuda.is_available(),
        logging_steps=10,
        save_strategy="epoch",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        tokenizer=tokenizer,
    )

    try:
        trainer.train()
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)
        print(f"Fine-tuned model saved to {output_dir}")
    except Exception as e:
        print(f"Fine-tuning failed: {str(e)}")


# --- XML Translator ---
class XMLTranslator:
    def __init__(self, model_name="finetuned_model"):
        self.device = 0 if torch.cuda.is_available() else -1
        try:
            self.translator = pipeline(
                "translation",
                model=model_name,
                device=self.device,
                batch_size=1
            )
        except Exception:
            print(f"Loading {model_name} failed, falling back to Helsinki-NLP/opus-mt-en-de")
            self.translator = pipeline(
                "translation",
                model="Helsinki-NLP/opus-mt-en-de",
                device=self.device,
                batch_size=1
            )

        self.translate_tags = {'title', 'head', 'p', 'note'}
        self.preserve_tags = {'xref', 'date', 'classCode', 'bibl'}

        self.term_patterns = {
            re.compile(r'\bOfficial Journal\b', re.I): 'Amtsblatt',
            re.compile(r'\bpaper edition\b', re.I): 'Papierausgabe',
            re.compile(r'\bdeemed authentic\b', re.I): 'verbindlich gelten'
        }

    def split_long_text(self, text, max_len=100):
        """Split text into chunks."""
        words = text.split()
        chunks = []
        current_chunk = []
        current_len = 0

        for word in words:
            word_len = len(word) + 1
            if current_len + word_len > max_len and current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = [word]
                current_len = word_len
            else:
                current_chunk.append(word)
                current_len += word_len

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    def translate_text(self, text):
        try:
            text = text.strip()
            if not text:
                return text

            max_chunk_len = 100
            if len(text) > max_chunk_len:
                chunks = self.split_long_text(text, max_chunk_len)
                translated_chunks = []
                for chunk in chunks:
                    if chunk:
                        result = self.translator(
                            chunk,
                            max_length=256,
                            truncation=True,
                            clean_up_tokenization_spaces=True,
                            num_beams=5
                        )[0]['translation_text']
                        result = re.sub(r'-{3,}', '', result)  # Remove excessive dashes
                        translated_chunks.append(result)
                result = " ".join(translated_chunks)
            else:
                result = self.translator(
                    text,
                    max_length=256,
                    truncation=True,
                    clean_up_tokenization_spaces=True,
                    num_beams=5
                )[0]['translation_text']
                result = re.sub(r'-{3,}', '', result)  # Remove excessive dashes

            for pattern, replacement in self.term_patterns.items():
                result = pattern.sub(replacement, result)
            return result
        except Exception as e:
            print(f"Translation failed for text '{text[:50]}...': {str(e)}")
            return text

    def process_xml(self, input_path, output_dir):
        try:
            tree = ET.parse(input_path)
            root = tree.getroot()

            self.update_language_attributes(root)

            for elem in root.iter():
                if elem.tag in self.translate_tags and elem.text:
                    self.process_element(elem)
                elif elem.tag in self.preserve_tags:
                    self.preserve_element(elem)

            original_name = input_path.name
            new_name = original_name.replace("-en.xml", "-de.xml")
            output_path = output_dir / new_name
            self.write_xml(tree, output_path)
            print(f"Processed: {input_path.name}")
        except Exception as e:
            print(f"Error processing {input_path.name}: {str(e)}")

    def process_element(self, elem):
        leading_ws = elem.text[:len(elem.text) - len(elem.text.lstrip())]
        trailing_ws = elem.text[len(elem.text.rstrip()):]
        content = elem.text.strip()
        if content:
            translated = self.translate_text(content)
            elem.text = f"{leading_ws}{translated}{trailing_ws}"

    def preserve_element(self, elem):
        if elem.text:
            elem.text = elem.text.strip()

    def update_language_attributes(self, root):
        for elem in [root, root.find('.//teiHeader')]:
            if elem is not None:
                elem.set('lang', 'de')

    def write_xml(self, tree, output_path):
        tree.write(
            output_path,
            encoding='utf-8',
            xml_declaration=True,
            method='xml',
            short_empty_elements=False
        )


def batch_translate_xml(input_dir="input_xml", output_dir="output_xml"):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    translator = XMLTranslator()

    xml_files = list(input_dir.glob("*-en.xml"))
    if not xml_files:
        print(f"No XML files found in {input_dir}")
        return

    print(f"Found {len(xml_files)} files to process")

    for idx, xml_file in enumerate(xml_files, 1):
        print(f"Processing file {idx}/{len(xml_files)}: {xml_file.name}")
        translator.process_xml(xml_file, output_dir)


def extract_text_from_xml_etree(filepath):
    """Extracts text from <p> and <head> tags using ElementTree."""
    try:
        tree = ET.parse(filepath)
        root = tree.getroot()

        # Collect text from <p> and <head> tags
        texts = [elem.text.strip() for elem in root.iter() if elem.tag in ('p', 'head') and elem.text and elem.text.strip()]
        return texts
    except ET.ParseError as e:
        print(f"❌ Error parsing XML file {filepath}: {e}")
        return []


class TranslationEvaluator:
    def __init__(self):
        self.metrics = {
            'bleu': load('bleu'),
            'meteor': load('meteor'),
            'ter': load('ter'),
            'comet_qe': load('comet')
        }

    def evaluate_file(self, ref_path, hyp_path, scs_path):
        ref_lines = extract_text_from_xml_etree(ref_path)
        hyp_lines = extract_text_from_xml_etree(hyp_path)
        scs_lines = extract_text_from_xml_etree(scs_path)

        aligned_refs, aligned_hyps, aligned_scs = self._align_lines(ref_lines, hyp_lines, scs_lines)

        if not aligned_refs:
            print("⚠️ No aligned lines found. Skipping evaluation.")
            return {}

        
        filtered_hyps = self._filter_none(aligned_hyps)
        filtered_refs = self._filter_none(aligned_refs)
        filtered_scs = self._filter_none(aligned_scs)
        formatted_refs = [[line] for line in filtered_refs]

        return self._calculate_metrics(filtered_hyps, filtered_refs, filtered_scs)

    def _align_lines(self, ref_lines, hyp_lines, scs_lines):
        min_len = min(len(ref_lines), len(hyp_lines), len(scs_lines))
        return ref_lines[:min_len], hyp_lines[:min_len], scs_lines[:min_len]

    def _filter_none(self, items):
        return [item if item is not None else "" for item in items]

    def _calculate_metrics(self, hypotheses, references, sources):
        results = {}

        try:
            results['bleu'] = self.metrics['bleu'].compute(
                predictions=hypotheses,
                references=references
            ).get('bleu', None)
        except Exception as e:
            print(f"BLEU metric computation failed: {e}")

        try:
            results['meteor'] = self.metrics['meteor'].compute(
                predictions=hypotheses,
                references=references
            ).get('meteor', None)
        except Exception as e:
            print(f"METEOR metric computation failed: {e}")

        try:
            results['ter'] = self.metrics['ter'].compute(
                predictions=hypotheses,
                references=references
            ).get('score', None)
        except Exception as e:
            print(f"TER metric computation failed: {e}")

        try:
            comet_output = self.metrics['comet_qe'].compute(
                predictions=hypotheses,
                references=[r[0] for r in references],
                sources=sources
            )

            results['comet_qe'] = comet_output.get('mean_score', None)
        except Exception as e:
            print(f"COMET QE computation failed: {e}")
            results['comet_qe'] = None


        return results





def generate_report(input_dir, output_dir, ref_dir, output_csv):
    evaluator = TranslationEvaluator()
    results = []

    output_path = Path(output_dir)
    print(f"📂 Scanning: {output_path.absolute()}")
    
    processed_files = []

    for hyp_file in output_path.glob('*.xml'):
        scs_file = Path(input_dir) / hyp_file.name.replace("-de.xml", "-en.xml")
        ref_file = Path(ref_dir) / hyp_file.name

        print(f"\n🔍 Processing:\n- Hypothesis: {hyp_file.name}\n- Reference: {ref_file.name}")

        if not ref_file.exists():
            print(f"⚠️ Missing reference file: {ref_file.name} - skipping")
            continue

        try:
            res = evaluator.evaluate_file(ref_file, hyp_file, scs_file)
            res['file'] = hyp_file.name
            results.append(res)
            print("✅ Metrics:", res)
            processed_files.append(hyp_file.name)

        except Exception as e:
            print(f"❌ Error processing {hyp_file.name}: {e}")
            continue
    if not results:
        print("No results collected")
        return pd.DataFrame()

    print(f"\nEvaluated {len(processed_files)} files: {', '.join(processed_files)}")
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"Saved report to: {output_csv}")
    return df


# --- Main Execution ---
def main():
    TRAIN_INPUT = Path("data/input")
    TRAIN_REF = Path("data/ref")
    INPUT_DIR = Path("input_xml")
    REF_DIR = Path("ref_xml")
    OUTPUT_DIR = Path("output_xml")
    RESULT_DIR = Path("result")
    DATA_DIR = Path("data")

    for d in [INPUT_DIR, REF_DIR, OUTPUT_DIR, RESULT_DIR, DATA_DIR, TRAIN_INPUT, TRAIN_REF]:
        d.mkdir(exist_ok=True)

    print("Preparing training data...")
    create_training_data(TRAIN_INPUT, TRAIN_REF)

    print("\nFine-tuning model...")
    fine_tune_model()

    print("\nTranslating XML files...")
    batch_translate_xml(INPUT_DIR, OUTPUT_DIR)

    print("\nGenerating evaluation report...")
    df = generate_report(INPUT_DIR, OUTPUT_DIR, REF_DIR, RESULT_DIR / "translation_quality.csv")


    if not df.empty:
        print("\nEvaluation Summary:")
        print(f"Average BLEU: {df['bleu'].mean():.4f}")
        print(f"Average METEOR: {df['meteor'].mean():.4f}")
        if 'ter' in df and df['ter'].notna().any():
            print(f"Average TER: {df['ter'].mean():.4f}")
        if 'comet_qe' in df and df['comet_qe'].notna().any():
            print(f"Average COMET-QE: {df['comet_qe'].mean():.4f}")


if __name__ == "__main__":
    main()