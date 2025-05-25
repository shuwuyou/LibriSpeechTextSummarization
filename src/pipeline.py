import pathlib
import sys
import yaml

# Import the updated modules
import audio_cleaning
import text_cleaning
import transcribe_speech
import ner_model
import summarizationt5

class Config:
    """
    A simple class to manage configuration paths and provide a root resolver.
    """
    def __init__(self, pipeline_file_path: pathlib.Path):
        # Assuming configs.yaml is at project_root/config/configs.yaml
        # and pipeline.py is at project_root/src/pipeline.py
        self.root = pipeline_file_path.resolve().parents[1]
        self.config_path = self.root / "config" / "configs.yaml"

        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

        with self.config_path.open("r") as f:
            self.cfg = yaml.safe_load(f)

    def R(self, p: str) -> pathlib.Path:
        """
        Resolves a relative path to an absolute path based on the project root.
        """
        pth = pathlib.Path(p)
        return (self.root / pth).resolve() if not pth.is_absolute() else pth

    def get(self, *keys, default=None):
        """
        Safely retrieve a value from the nested configuration dictionary.
        Example: config.get("audio_cleaning", "input_path")
        """
        current = self.cfg
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return default
        return current

def run_full_pipeline():
    """
    Orchestrates the entire audio processing pipeline by calling
    functions from imported modules.
    """
    print("🚀 Starting the Audio Processing Pipeline...")

    # Initialize configuration
    pipeline_file_path = pathlib.Path(__file__)
    try:
        config = Config(pipeline_file_path)
        print(f"Loaded configuration from: {config.config_path}")
    except FileNotFoundError as e:
        print(f"❌ Pipeline setup failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error loading configuration: {e}")
        sys.exit(1)


    # Step 1: Audio Cleaning
    print("\n--- Step 1: Audio Cleaning ---")
    try:
        audio_cleaning.run_audio_cleaning(config.config_path)
        print("✅ Audio cleaning completed.")
    except Exception as e:
        print(f"❌ Audio cleaning failed: {e}")
        sys.exit(1)

    # Step 2: Speech Transcription
    print("\n--- Step 2: Speech Transcription ---")
    try:
        transcribe_speech.run_transcription(config.config_path)
        print("✅ Speech transcription completed.")
    except Exception as e:
        print(f"❌ Speech transcription failed: {e}")
        sys.exit(1)

    # Step 3: Text Cleaning
    print("\n--- Step 3: Text Cleaning ---")
    try:
        text_cleaning.run_text_cleaning(config.config_path)
        print("✅ Text cleaning completed.")
    except Exception as e:
        print(f"❌ Text cleaning failed: {e}")
        sys.exit(1)

    # Step 4: Named Entity Recognition (NER)
    print("\n--- Step 4: Named Entity Recognition (NER) ---")
    try:
        ner_model.run_ner_extraction(config.config_path)
        print("✅ Named Entity Recognition completed.")
    except Exception as e:
        print(f"❌ Named Entity Recognition failed: {e}")
        sys.exit(1)

    # Step 5: Summarization
    print("\n--- Step 5: Text Summarization ---")
    try:
        summarizationt5.run_summarization(config.config_path)
        print("✅ Text summarization completed.")
    except Exception as e:
        print(f"❌ Text summarization failed: {e}")
        sys.exit(1)

    print("\n🎉 Audio Processing Pipeline Finished Successfully!")

if __name__ == "__main__":
    # To run this pipeline, ensure your project structure is:
    # project_root/
    # ├── config/
    # │   └── configs.yaml
    # └── src/
    #     ├── audio_cleaning.py
    #     ├── text_cleaning.py
    #     ├── transcribe_speech.py
    #     ├── ner_model.py
    #     ├── summarizationt5.py
    #     └── pipeline.py
    run_full_pipeline()
