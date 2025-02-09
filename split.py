import pathlib
import shutil
import random

def split_dataset(dataset_path, output_path, train_ratio, val_ratio):
    dataset_path = pathlib.Path(dataset_path)
    output_path = pathlib.Path(output_path)
    
    # Creazione delle cartelle di output
    (output_path / "train").mkdir(parents=True, exist_ok=True)
    (output_path / "val").mkdir(parents=True, exist_ok=True)
    (output_path / "test").mkdir(parents=True, exist_ok=True)
    
    for class_dir in dataset_path.iterdir():
        if class_dir.is_dir():
            images = list(class_dir.glob("*.jpg"))
            
            # Mescolamento casuale
            random.shuffle(images)
            
            # Calcolo delle divisioni
            train_cutoff = int(len(images) * train_ratio)
            val_cutoff = int(len(images) * (train_ratio + val_ratio))
            
            # Funzione per copiare immagini
            def copy_images(imgs, dest_dir):
                dest_dir.mkdir(parents=True, exist_ok=True)
                for img in imgs:
                    shutil.copy(img, dest_dir / img.name)
            
            # Distribuzione delle immagini
            copy_images(images[:train_cutoff], output_path / "train" / class_dir.name)
            copy_images(images[train_cutoff:val_cutoff], output_path / "val" / class_dir.name)
            copy_images(images[val_cutoff:], output_path / "test" / class_dir.name)
    
    print("Dataset split complete!")

if __name__ == "__main__":
    dataset_path = "/Users/edoardopavan/Desktop/Universita/Magistrale/Vision/testProgetto/hagrid-sample/hagrid-sample-500k-384p/hagrid_500k"  # Sostituisci con il percorso del dataset
    output_path = "/Users/edoardopavan/Desktop/Universita/Magistrale/Vision/testProgetto/hagrid-sample/hagrid-sample-500k-384p/split"  # Sostituisci con il percorso di output
    
    train_ratio = 0.8
    val_ratio = 0.1
    test_ratio = 0.1
    
    split_dataset(dataset_path, output_path, train_ratio, val_ratio)