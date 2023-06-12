python tools/make_masks.py -i ./data/restor-tcd-oam/images -o ./data/restor-tcd-oam/downsampled/256 --annotations ./data/restor-tcd-oam/train_20221010_noempty.json -s 256 --parallel --visualise --resize_instances --binary
python tools/make_masks.py -i ./data/restor-tcd-oam/images -o ./data/restor-tcd-oam/downsampled/512 --annotations ./data/restor-tcd-oam/train_20221010_noempty.json -s 512 --parallel --visualise --resize_instances --binary
python tools/make_masks.py -i ./data/restor-tcd-oam/images -o ./data/restor-tcd-oam/downsampled/1024 --annotations ./data/restor-tcd-oam/train_20221010_noempty.json -s 1024 --parallel --visualise --resize_instances --binary
python tools/make_masks.py -i ./data/restor-tcd-oam/images -o ./data/restor-tcd-oam/downsampled/2048 --annotations ./data/restor-tcd-oam/train_20221010_noempty.json -s 2048 --parallel --visualise --resize_instances --binary

python tools/make_masks.py -i ./data/restor-tcd-oam/images -o ./data/restor-tcd-oam/downsampled/256 --annotations ./data/restor-tcd-oam/val_20221010_noempty.json -s 256 --parallel --visualise --resize_instances --binary
python tools/make_masks.py -i ./data/restor-tcd-oam/images -o ./data/restor-tcd-oam/downsampled/512 --annotations ./data/restor-tcd-oam/val_20221010_noempty.json -s 512 --parallel --visualise --resize_instances --binary
python tools/make_masks.py -i ./data/restor-tcd-oam/images -o ./data/restor-tcd-oam/downsampled/1024 --annotations ./data/restor-tcd-oam/val_20221010_noempty.json -s 1024 --parallel --visualise --resize_instances --binary
python tools/make_masks.py -i ./data/restor-tcd-oam/images -o ./data/restor-tcd-oam/downsampled/2048 --annotations ./data/restor-tcd-oam/val_20221010_noempty.json -s 2048 --parallel --visualise --resize_instances --binary

python tools/make_masks.py -i ./data/restor-tcd-oam/images -o ./data/restor-tcd-oam/downsampled/256 --annotations ./data/restor-tcd-oam/test_20221010_noempty.json -s 256 --parallel --visualise --resize_instances --binary
python tools/make_masks.py -i ./data/restor-tcd-oam/images -o ./data/restor-tcd-oam/downsampled/512 --annotations ./data/restor-tcd-oam/test_20221010_noempty.json -s 512 --parallel --visualise --resize_instances --binary
python tools/make_masks.py -i ./data/restor-tcd-oam/images -o ./data/restor-tcd-oam/downsampled/1024 --annotations ./data/restor-tcd-oam/test_20221010_noempty.json -s 1024 --parallel --visualise --resize_instances --binary
python tools/make_masks.py -i ./data/restor-tcd-oam/images -o ./data/restor-tcd-oam/downsampled/2048 --annotations ./data/restor-tcd-oam/test_20221010_noempty.json -s 2048 --parallel --visualise --resize_instances --binary
