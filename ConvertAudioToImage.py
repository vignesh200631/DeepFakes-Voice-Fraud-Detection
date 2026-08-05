import os
import warnings
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# ============================================================
# PATHS
# ============================================================

INPUT_DIR = r"D:\Project Folder\Mini Project\Datasets\Audio Datasets\Raw Dataset"

OUTPUT_DIR = r"D:\Project Folder\Mini Project\Datasets\Audio Datasets\Spectrogram Dataset"

FAILED_FILE = r"D:\Project Folder\Mini Project\Datasets\Audio Datasets\failed_audio.txt"

CLASSES = ["AI Voice", "Human Voice"]


# ============================================================
# AUDIO SETTINGS
# ============================================================

SAMPLE_RATE = 16000
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512

SUPPORTED_FORMATS = (".wav", ".mp3", ".flac")


# ============================================================
# SUPPRESS LIBROSA WARNINGS
# ============================================================

warnings.filterwarnings(
    "ignore",
    message="PySoundFile failed"
)

warnings.filterwarnings(
    "ignore",
    category=FutureWarning
)


# ============================================================
# SAVE FAILED FILE
# ============================================================

def save_failed_file(audio_path, error):

    try:
        with open(
            FAILED_FILE,
            "a",
            encoding="utf-8"
        ) as f:

            f.write(
                f"{audio_path} | {type(error).__name__}: {error}\n"
            )

    except Exception:
        pass


# ============================================================
# CONVERT SINGLE AUDIO FILE
# ============================================================

def create_spectrogram(audio_path, output_path):

    try:

        # ----------------------------------------------------
        # Basic file validation
        # ----------------------------------------------------

        if not os.path.exists(audio_path):
            raise FileNotFoundError("File does not exist")

        file_size = os.path.getsize(audio_path)

        # Very tiny files are usually invalid
        if file_size < 1000:
            raise ValueError(
                f"File too small ({file_size} bytes)"
            )

        # ----------------------------------------------------
        # Load audio
        # ----------------------------------------------------

        y, sr = librosa.load(
            audio_path,
            sr=SAMPLE_RATE,
            mono=True
        )

        # ----------------------------------------------------
        # Validate decoded audio
        # ----------------------------------------------------

        if y is None:
            raise ValueError("Audio could not be decoded")

        if len(y) == 0:
            raise ValueError("Empty audio")

        if not np.all(np.isfinite(y)):
            raise ValueError(
                "Audio contains NaN/Infinity values"
            )

        # ----------------------------------------------------
        # Create Mel Spectrogram
        # ----------------------------------------------------

        mel = librosa.feature.melspectrogram(
            y=y,
            sr=sr,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            n_mels=N_MELS,
            power=2.0
        )

        # ----------------------------------------------------
        # Convert to Log-Mel
        # ----------------------------------------------------

        log_mel = librosa.power_to_db(
            mel,
            ref=np.max
        )

        # ----------------------------------------------------
        # Create spectrogram image
        # ----------------------------------------------------

        fig = plt.figure(
            figsize=(3, 3),
            dpi=128,
            frameon=False
        )

        ax = fig.add_axes(
            [0, 0, 1, 1]
        )

        ax.axis("off")

        librosa.display.specshow(
            log_mel,
            sr=sr,
            hop_length=HOP_LENGTH,
            cmap="viridis",
            ax=ax
        )

        # ----------------------------------------------------
        # Save PNG
        # ----------------------------------------------------

        fig.savefig(
            output_path,
            bbox_inches=None,
            pad_inches=0
        )

        plt.close(fig)

        return True

    except KeyboardInterrupt:

        # Don't treat Ctrl+C as corrupted audio
        plt.close("all")

        raise

    except Exception as error:

        plt.close("all")

        save_failed_file(
            audio_path,
            error
        )

        return False


# ============================================================
# CONVERT COMPLETE DATASET
# ============================================================

def convert_dataset():

    print()
    print("=" * 55)
    print("     AUDIO TO LOG-MEL SPECTROGRAM CONVERTER")
    print("=" * 55)

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Remove previous failed file list
    # --------------------------------------------------------

    if os.path.exists(FAILED_FILE):

        try:
            os.remove(FAILED_FILE)

        except Exception:
            pass

    # --------------------------------------------------------
    # Global statistics
    # --------------------------------------------------------

    total_files = 0
    total_converted = 0
    total_skipped = 0
    total_failed = 0


    # ========================================================
    # PROCESS EACH CLASS
    # ========================================================

    for class_name in CLASSES:

        input_folder = os.path.join(
            INPUT_DIR,
            class_name
        )

        output_folder = os.path.join(
            OUTPUT_DIR,
            class_name
        )

        os.makedirs(
            output_folder,
            exist_ok=True
        )

        # ----------------------------------------------------
        # Check input folder
        # ----------------------------------------------------

        if not os.path.isdir(input_folder):

            print()
            print(
                f"Folder not found: {input_folder}"
            )

            continue


        # ----------------------------------------------------
        # Find supported audio
        # ----------------------------------------------------

        audio_files = [

            file

            for file in os.listdir(input_folder)

            if file.lower().endswith(
                SUPPORTED_FORMATS
            )
        ]


        # Sort files
        audio_files.sort()


        # ----------------------------------------------------
        # Count formats
        # ----------------------------------------------------

        wav_count = sum(
            file.lower().endswith(".wav")
            for file in audio_files
        )

        mp3_count = sum(
            file.lower().endswith(".mp3")
            for file in audio_files
        )

        flac_count = sum(
            file.lower().endswith(".flac")
            for file in audio_files
        )


        print()
        print("-" * 55)

        print(
            f"Class : {class_name}"
        )

        print("-" * 55)

        print(
            f"Total audio : {len(audio_files)}"
        )

        print(
            f"WAV         : {wav_count}"
        )

        print(
            f"MP3         : {mp3_count}"
        )

        print(
            f"FLAC        : {flac_count}"
        )

        print("-" * 55)


        total_files += len(audio_files)


        # ----------------------------------------------------
        # Class statistics
        # ----------------------------------------------------

        converted = 0
        skipped = 0
        failed = 0


        # ====================================================
        # CONVERT FILES
        # ====================================================

        try:

            for audio_file in tqdm(
                audio_files,
                desc=f"Converting {class_name}",
                unit="file"
            ):

                audio_path = os.path.join(
                    input_folder,
                    audio_file
                )


                # --------------------------------------------
                # Generate output PNG filename
                # --------------------------------------------

                filename_without_extension = (
                    os.path.splitext(
                        audio_file
                    )[0]
                )

                image_name = (
                    filename_without_extension
                    + ".png"
                )


                output_path = os.path.join(
                    output_folder,
                    image_name
                )


                # ============================================
                # ALREADY CONVERTED
                # ============================================

                if os.path.exists(output_path):

                    skipped += 1

                    continue


                # ============================================
                # CONVERT
                # ============================================

                success = create_spectrogram(
                    audio_path,
                    output_path
                )


                if success:

                    converted += 1

                else:

                    failed += 1


        except KeyboardInterrupt:

            print()
            print()
            print("Conversion stopped manually.")
            print("Don't worry: existing PNG files are saved.")
            print("Run the program again to continue.")

            return


        # ----------------------------------------------------
        # Update totals
        # ----------------------------------------------------

        total_converted += converted
        total_skipped += skipped
        total_failed += failed


        # ----------------------------------------------------
        # Print class result
        # ----------------------------------------------------

        print()

        print(
            f"{class_name} completed:"
        )

        print(
            f"  New converted    : {converted}"
        )

        print(
            f"  Already converted: {skipped}"
        )

        print(
            f"  Failed/corrupted : {failed}"
        )


    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print()
    print("=" * 55)
    print("                 CONVERSION COMPLETE")
    print("=" * 55)

    print(
        f"Total audio files   : {total_files}"
    )

    print(
        f"New images created  : {total_converted}"
    )

    print(
        f"Already converted   : {total_skipped}"
    )

    print(
        f"Failed/corrupted    : {total_failed}"
    )

    print("=" * 55)

    print()
    print("Spectrogram folder:")
    print(OUTPUT_DIR)


    # --------------------------------------------------------
    # Failed files
    # --------------------------------------------------------

    if total_failed > 0:

        print()
        print("Failed file list:")
        print(FAILED_FILE)

    else:

        print()
        print("No corrupted audio files found.")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    convert_dataset()