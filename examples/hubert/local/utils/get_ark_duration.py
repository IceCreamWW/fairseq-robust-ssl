import argparse
import kaldiio
import pdb

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--wav_scp", help="path to wav.scp", required=True)
    args = parser.parse_args()

    with open(args.wav_scp) as ifp:
        for line in ifp:
            uttid, audio = line.strip().split(maxsplit=1)
            rate, wave = kaldiio.load_mat(audio)
            frames = wave.shape[0]
            print(f"{uttid} {frames}", flush=True)
            
