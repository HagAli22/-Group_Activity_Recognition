import os
import sys
import pickle
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import random
sys.path.append('D:/pycharm project/slidesdeep/15 Final Project/Group-Activity-Recognition')

from data.boxinfo import BoxInfo


def load_tracking_annot(path):
    with open(path, 'r') as file:
        player_boxes = {idx: [] for idx in range(12)}
        frame_boxes_dct = {}

        for idx, line in enumerate(file):
            box_info = BoxInfo(line)

            if box_info.player_ID > 11:
                continue
            player_boxes[box_info.player_ID].append(box_info)

        for player_ID, boxes_info in player_boxes.items():
            boxes_info = boxes_info[5:]
            boxes_info = boxes_info[:-6]

            for box_info in boxes_info:
                if box_info.frame_ID not in frame_boxes_dct:
                    frame_boxes_dct[box_info.frame_ID] = []
                frame_boxes_dct[box_info.frame_ID].append(box_info)

        return frame_boxes_dct   # 9 frames per clip   , each frame has 12 players


def load_video_annot(video_annot):
    with open(video_annot, 'r') as file:
        clip_category_dct = {}

        for line in file:
            items = line.strip().split(' ')[:2]
            clip_dir = items[0].replace('.jpg', '')
            clip_category_dct[clip_dir] = items[1]

        return clip_category_dct


def load_volleyball_dataset(videos_root, annot_root):
    videos_dirs = os.listdir(videos_root)
    videos_dirs.sort()

    videos_annot = {}

    for idx, video_dir in enumerate(videos_dirs):
        video_dir_path = os.path.join(videos_root, video_dir)

        if not os.path.isdir(video_dir_path):
            continue

        print(f'{idx}/{len(videos_dirs)} - Processing Dir {video_dir_path}')

        video_annot = os.path.join(video_dir_path, 'annotations.txt')
        clip_category_dct = load_video_annot(video_annot)

        clips_dir = os.listdir(video_dir_path)
        clips_dir.sort()

        clip_annot = {}

        for clip_dir in clips_dir:
            clip_dir_path = os.path.join(video_dir_path, clip_dir)

            if not os.path.isdir(clip_dir_path):
                continue

            assert clip_dir in clip_category_dct

            annot_file = os.path.join(annot_root, video_dir, clip_dir, f'{clip_dir}.txt')
            frame_boxes_dct = load_tracking_annot(annot_file)

            clip_annot[clip_dir] = {
                'category': clip_category_dct[clip_dir],
                'frame_boxes_dct': frame_boxes_dct
            }

        videos_annot[video_dir] = clip_annot

    return videos_annot


class VolleyballDataset(Dataset):
    """
    General purpose volleyball dataset loader for different baseline models
    """

    def __init__(self,
                 dataset_root,
                 mode='B1',  # B1, B3, B4, B5, B6, B7, B8
                 split='train',  # train, val, test
                 transform=None,
                 sequence_length=9,
                 middle_frame_only=False,
                 load_players=False,
                 team_split=False):

        self.dataset_root = dataset_root
        self.mode = mode
        self.split = split
        self.transform = transform
        self.sequence_length = sequence_length
        self.middle_frame_only = middle_frame_only
        self.load_players = load_players
        self.team_split = team_split

        # Load annotations
        annot_path = os.path.join(dataset_root, 'annot_all.pkl')
        if os.path.exists(annot_path):
            with open(annot_path, 'rb') as f:
                self.videos_annot = pickle.load(f)
        else:
            # Create annotations if not exist
            videos_root = os.path.join(dataset_root, 'videos')
            annot_root = os.path.join(dataset_root, 'volleyball_tracking_annotation')
            self.videos_annot = load_volleyball_dataset(videos_root, annot_root)

            # Save for future use
            with open(annot_path, 'wb') as f:
                pickle.dump(self.videos_annot, f)

        # Class mapping
        self.action_classes = ['r_set', 'r_spike', 'r-pass', 'r_winpoint',
                               'l_set', 'l-spike', 'l-pass', 'l_winpoint']
        self.individual_classes = ['blocking', 'digging', 'falling', 'jumping',
                                   'moving', 'setting', 'spiking', 'standing', 'waiting']

        self.action_to_idx = {action: idx for idx, action in enumerate(self.action_classes)}
        self.individual_to_idx = {action: idx for idx, action in enumerate(self.individual_classes)}

        # Data split (you may want to modify this based on your actual split)
        self.video_splits = self._create_splits()
        self.data_samples = self._prepare_samples()

    def _create_splits(self):
        """Create train/val/test splits"""
        videos = list(self.videos_annot.keys())
        videos.sort()

        # Simple split: 70% train, 15% val, 15% test
        n_videos = len(videos)
        n_train = int(0.7 * n_videos)
        n_val = int(0.15 * n_videos)

        splits = {
            'train': videos[:n_train],
            'val': videos[n_train:n_train + n_val],
            'test': videos[n_train + n_val:]
        }

        return splits

    def _prepare_samples(self):
        """Prepare data samples based on mode and split"""
        samples = []

        for video_id in self.video_splits[self.split]:
            video_data = self.videos_annot[video_id]

            for clip_id, clip_data in video_data.items():
                clip_path = os.path.join(self.dataset_root, 'videos', video_id, clip_id)

                if not os.path.exists(clip_path):
                    continue

                # Get all frames in clip
                frames = [f for f in os.listdir(clip_path) if f.endswith('.jpg')]
                frames.sort()

                if len(frames) < self.sequence_length:
                    continue

                sample = {
                    'video_id': video_id,
                    'clip_id': clip_id,
                    'clip_path': clip_path,
                    'category': clip_data['category'],
                    'frame_boxes_dct': clip_data['frame_boxes_dct'],
                    'frames': frames
                }

                samples.append(sample)

        return samples

    def _load_image(self, image_path):
        """Load and preprocess image"""
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            image = Image.fromarray(image)
            image = self.transform(image)
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        return image

    def _crop_player(self, image, box, target_size=(224, 224)):
        """Crop player from image based on bounding box"""
        if isinstance(image, torch.Tensor):
            # Convert tensor to numpy for cropping
            image_np = image.permute(1, 2, 0).numpy()
            if image_np.max() <= 1.0:
                image_np = (image_np * 255).astype(np.uint8)
        else:
            image_np = image

        x1, y1, x2, y2 = box
        crop = image_np[y1:y2, x1:x2]

        if crop.size == 0:
            crop = np.zeros((target_size[0], target_size[1], 3), dtype=np.uint8)
        else:
            crop = cv2.resize(crop, target_size)

        crop = torch.from_numpy(crop).permute(2, 0, 1).float() / 255.0
        return crop

    def _get_middle_frame_data(self, sample):
        """Get middle frame data for B1 baseline"""
        frames = sample['frames']
        middle_idx = len(frames) // 2
        middle_frame = frames[middle_idx]

        image_path = os.path.join(sample['clip_path'], middle_frame)
        image = self._load_image(image_path)

        label = self.action_to_idx[sample['category']]

        return image, label

    def _get_sequence_data(self, sample):
        """Get sequence data for temporal models"""
        frames = sample['frames']

        # Sample sequence_length frames
        if len(frames) >= self.sequence_length:
            start_idx = (len(frames) - self.sequence_length) // 2
            selected_frames = frames[start_idx:start_idx + self.sequence_length]
        else:
            selected_frames = frames + [frames[-1]] * (self.sequence_length - len(frames))

        sequence = []
        for frame in selected_frames:
            image_path = os.path.join(sample['clip_path'], frame)
            image = self._load_image(image_path)
            sequence.append(image)

        sequence = torch.stack(sequence)
        label = self.action_to_idx[sample['category']]

        return sequence, label

    def _get_players_data(self, sample):
        """Get players data for player-level models"""
        frames = sample['frames']
        frame_boxes_dct = sample['frame_boxes_dct']

        # Sample sequence_length frames
        if len(frames) >= self.sequence_length:
            start_idx = (len(frames) - self.sequence_length) // 2
            selected_frames = frames[start_idx:start_idx + self.sequence_length]
        else:
            selected_frames = frames + [frames[-1]] * (self.sequence_length - len(frames))

        players_sequence = []

        for frame in selected_frames:
            frame_num = int(frame.split('.')[0])
            image_path = os.path.join(sample['clip_path'], frame)
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            frame_players = []

            if frame_num in frame_boxes_dct:
                boxes_info = frame_boxes_dct[frame_num]

                if self.team_split:
                    # Separate teams (B8 baseline)
                    team1_crops = []
                    team2_crops = []

                    for box_info in boxes_info:
                        if box_info.lost or box_info.generated:
                            continue

                        crop = self._crop_player(image, box_info.box)

                        # Simple team assignment based on player_ID
                        if box_info.player_ID < 6:
                            team1_crops.append(crop)
                        else:
                            team2_crops.append(crop)

                    # Pad to 6 players per team
                    while len(team1_crops) < 6:
                        team1_crops.append(torch.zeros(3, 224, 224))
                    while len(team2_crops) < 6:
                        team2_crops.append(torch.zeros(3, 224, 224))

                    team1_crops = torch.stack(team1_crops[:6])
                    team2_crops = torch.stack(team2_crops[:6])
                    frame_players = [team1_crops, team2_crops]

                else:
                    # All players together
                    for box_info in boxes_info:
                        if box_info.lost or box_info.generated:
                            continue
                        crop = self._crop_player(image, box_info.box)
                        frame_players.append(crop)

                    # Pad to maximum players
                    while len(frame_players) < 12:
                        frame_players.append(torch.zeros(3, 224, 224))

                    frame_players = torch.stack(frame_players[:12])

            else:
                # No players detected, use zero tensors
                if self.team_split:
                    team1_crops = torch.zeros(6, 3, 224, 224)
                    team2_crops = torch.zeros(6, 3, 224, 224)
                    frame_players = [team1_crops, team2_crops]
                else:
                    frame_players = torch.zeros(12, 3, 224, 224)

            players_sequence.append(frame_players)

        if self.team_split:
            # Separate team sequences
            team1_sequence = torch.stack([frame[0] for frame in players_sequence])
            team2_sequence = torch.stack([frame[1] for frame in players_sequence])
            players_data = [team1_sequence, team2_sequence]
        else:
            players_data = torch.stack(players_sequence)

        label = self.action_to_idx[sample['category']]
        return players_data, label

    def __len__(self):
        return len(self.data_samples)

    def __getitem__(self, idx):
        sample = self.data_samples[idx]

        if self.mode == 'B1' and self.middle_frame_only:
            data=[]
            frames=sample['frames']
            clip_path=sample['clip_path']
            label = self.action_to_idx[sample['category']]

            for idx,frame in frames:
                frame_path=os.path.join(clip_path,frame)
                data.append({
                    'frame':frame_path,
                    'label':label
                })


            # B1: Middle frame only
            return self._get_middle_frame_data(sample)

        elif self.mode in ['B4', 'B6']:
            # B4, B6: Sequence of frames
            return self._get_sequence_data(sample)

        elif self.mode in ['B3']:
            # B3: Players in middle frame
            frames = sample['frames']
            middle_idx = len(frames) // 2
            middle_frame = frames[middle_idx]
            frame_num = int(middle_frame.split('.')[0])

            image_path = os.path.join(sample['clip_path'], middle_frame)
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            players = []
            frame_boxes_dct = sample['frame_boxes_dct']

            if frame_num in frame_boxes_dct:
                boxes_info = frame_boxes_dct[frame_num]
                for box_info in boxes_info:
                    if box_info.lost or box_info.generated:
                        continue
                    crop = self._crop_player(image, box_info.box)
                    players.append(crop)

            # Pad to maximum players
            while len(players) < 12:
                players.append(torch.zeros(3, 224, 224))

            players = torch.stack(players[:12])
            label = self.action_to_idx[sample['category']]
            return players, label

        elif self.mode in ['B5', 'B7']:
            # B5, B7: Player sequences
            return self._get_players_data(sample)

        elif self.mode == 'B8':
            # B8: Team-separated player sequences
            return self._get_players_data(sample)

        else:
            # Default: return sequence data
            return self._get_sequence_data(sample)


def create_data_loaders(dataset_root,
                        mode='B1',
                        batch_size=32,
                        num_workers=0,
                        sequence_length=9,
                        middle_frame_only=False,
                        team_split=False):
    """
    Create train, val, and test data loaders for specified baseline
    """

    # Define transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Create datasets
    train_dataset = VolleyballDataset(
        dataset_root=dataset_root,
        mode=mode,
        split='train',
        transform=train_transform,
        sequence_length=sequence_length,
        middle_frame_only=middle_frame_only,
        team_split=team_split
    )

    val_dataset = VolleyballDataset(
        dataset_root=dataset_root,
        mode=mode,
        split='val',
        transform=val_transform,
        sequence_length=sequence_length,
        middle_frame_only=middle_frame_only,
        team_split=team_split
    )

    test_dataset = VolleyballDataset(
        dataset_root=dataset_root,
        mode=mode,
        split='test',
        transform=val_transform,
        sequence_length=sequence_length,
        middle_frame_only=middle_frame_only,
        team_split=team_split
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader


# Usage examples for each baseline:


class get_B1_loaders(Dataset):
    def __init__(self, videos_path,annot_path, split, transform=None):
        self.samples = []
        self.transform = transform
        self.categories_dct = {
            'l-pass': 0,
            'r-pass': 1,
            'l-spike': 2,
            'r_spike': 3,
            'l_set': 4,
            'r_set': 5,
            'l_winpoint': 6,
            'r_winpoint': 7
        }

        self.data=[]
        # print("File exists:", os.path.exists(annot_path))
        with open(annot_path,'rb')as file:
            
            videos_annot=pickle.load(file)        


        for idx in split:
            # print("6")
            video_annot=videos_annot[str(idx)]

            for clip in video_annot.keys():
                frames_data = video_annot[str(clip)]['frame_boxes_dct']
                category = video_annot[str(clip)]['category']
                dir_frames = list(video_annot[str(clip)]['frame_boxes_dct'].items())

                for frame_id,boxes in dir_frames:
                    #if str(clip)==str(frame_id):
                    self.data.append(
                        {
                            'frame_path':f'{videos_path}/{str(idx)}/{str(clip)}/{frame_id}.jpg',
                            'category':category
                        }
                    )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample=self.data[idx]
        num_classes=len(self.categories_dct)
        label=torch.zeros(num_classes)
        label[self.categories_dct[sample['category']]]=1
        # label = self.categories_dct[sample['category']]

        frame = Image.open(sample['frame_path']).convert('RGB')

        if self.transform:
            frame = self.transform(frame)

        return frame, label
    
class get_B3_A_loaders(Dataset):
    def __init__(self, videos_path, annot_path, split, transform=None, logger=None):
        self.samples = []
        self.transform = transform
        self.categories_dct = {
            'waiting': 0,
            'setting': 1,
            'digging': 2,
            'falling': 3,
            'spiking': 4,
            'blocking': 5,
            'jumping': 6,
            'moving': 7,
            'standing': 8
        }

        self.data = []
        
        
        try:
            with open(annot_path, 'rb') as file:
                videos_annot = pickle.load(file)
        except Exception as e:
            raise

        box_count = 0
        for idx in split:
            video_annot = videos_annot[str(idx)]

            for clip in video_annot.keys():
                frames_data = video_annot[str(clip)]['frame_boxes_dct']
                dir_frames = list(video_annot[str(clip)]['frame_boxes_dct'].items())

                for frame_id, boxes in dir_frames:
                    image_path = f'{videos_path}/{str(idx)}/{str(clip)}/{frame_id}.jpg'
                    image_path = os.path.join(image_path)

                    for box_info in boxes:
                        x1, y1, x2, y2 = box_info.box
                        category = box_info.category
                        box_count += 1

                        self.data.append(
                            {
                                'frame_path': f'{videos_path}/{str(idx)}/{str(clip)}/{frame_id}.jpg',
                                'box': (x1, y1, x2, y2),
                                'category': category
                            }
                        )
        
        

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        num_classes = len(self.categories_dct)
        labels = torch.zeros(num_classes)
        category = self.categories_dct[sample['category']]

        labels[self.categories_dct[sample['category']]] = 1

        image = Image.open(sample['frame_path']).convert('RGB')
        x1, y1, x2, y2 = sample['box']
        cropped_image = image.crop((x1, y1, x2, y2))
    
        if self.transform:
            cropped_image = self.transform(image=np.array(cropped_image))['image']

        return cropped_image, labels


class get_B3_B_loaders(Dataset):
    def __init__(self, videos_path,annot_path, split, transform=None):
        self.samples = []
        self.transform = transform
        self.categories_dct = {
            'l-pass': 0,
            'r-pass': 1,
            'l-spike': 2,
            'r_spike': 3,
            'l_set': 4,
            'r_set': 5,
            'l_winpoint': 6,
            'r_winpoint': 7
        }
        # print("Available videos in data:", list(dataset_dict.keys()))
        # print("Available labels in labels_dict:", list(label_dict.keys()))

        self.data=[]
        with open(annot_path,'rb')as file:
            videos_annot=pickle.load(file)


        for idx in split:
            video_annot=videos_annot[str(idx)]

            for clip in video_annot.keys():
                frames_data = video_annot[str(clip)]['frame_boxes_dct']
                category = video_annot[str(clip)]['category']
                dir_frames = list(video_annot[str(clip)]['frame_boxes_dct'].items())

                for frame_id,boxes in dir_frames:
                    image_path=f'{videos_path}/{str(idx)}/{str(clip)}/{frame_id}.jpg'

                    self.data.append(
                        {
                            'frame_path':f'{videos_path}/{str(idx)}/{str(clip)}/{frame_id}.jpg',
                            'image_path': image_path,
                            'boxes': boxes,
                            'category':category
                        }
                    )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample=self.data[idx]
        num_classes=len(self.categories_dct)
        labels=torch.zeros(num_classes)

        labels[self.categories_dct[sample['category']]]=1


        image_path=sample['image_path']
        image_path=os.path.join(image_path)
        image=Image.open(image_path).convert('RGB')

        processed_cropred_images=[]
        boxes=sample['boxes']
        T=12
        for box_info in boxes:
            x1,y1,x2,y2=box_info.box
            cropred_image=image.crop((x1,y1,x2,y2))
            cropred_image=self.transform(image=np.array(cropred_image))['image']
            # cropred_image=self.transform(image=cropred_image)['image']
            processed_cropred_images.append(cropred_image)

        while len(processed_cropred_images) < T:
            processed_cropred_images.append(torch.zeros(3, 224, 224))


        processed_cropred_images=torch.stack(processed_cropred_images)

        return processed_cropred_images, labels

class get_B4_loaders(Dataset):
    def __init__(self, videos_path,annot_path, split, transform=None):
        self.samples = []
        self.transform = transform
        self.categories_dct = {
            'l-pass': 0,
            'r-pass': 1,
            'l-spike': 2,
            'r_spike': 3,
            'l_set': 4,
            'r_set': 5,
            'l_winpoint': 6,
            'r_winpoint': 7
        }
        # print("Available videos in data:", list(dataset_dict.keys()))
        # print("Available labels in labels_dict:", list(label_dict.keys()))

        self.data=[]
        with open(annot_path,'rb')as file:
            videos_annot=pickle.load(file)
        print("done loading annotations")


        for idx in split:
            video_annot=videos_annot[str(idx)]

            for clip in video_annot.keys():
                frames_data = video_annot[str(clip)]['frame_boxes_dct']
                category = video_annot[str(clip)]['category']
                dir_frames = list(video_annot[str(clip)]['frame_boxes_dct'].items())
                clip_path = f'{videos_path}/{str(idx)}/{str(clip)}'
                # seq_of_frames=[]
                # for frame_id,boxes in dir_frames:
                #     image_path = f'{videos_path}/{str(idx)}/{str(clip)}/{frame_id}.jpg'
                #     image_path = os.path.join(image_path)
                #     image = Image.open(image_path).convert('RGB')

                #     image = self.transform(image=np.array(image))['image']
                #     seq_of_frames.append(image)

                # seq_of_frames = torch.stack(seq_of_frames)

                self.data.append(
                    {
                        # 'frame_path':f'{videos_path}/{str(idx)}/{str(clip)}/{frame_id}.jpg',
                        # 'seq_of_frames':seq_of_frames,
                        'dir_frames': dir_frames,
                        'clip_path': clip_path,
                        'category':category
                    }
                )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample=self.data[idx]
        num_classes=len(self.categories_dct)
        labels=torch.zeros(num_classes)
        labels[self.categories_dct[sample['category']]]=1

        # seq_of_frames=sample['seq_of_frames']
        dir_frames = sample['dir_frames']
        clip_path = sample['clip_path']
        seq_of_frames=[]
        for frame_id,boxes in dir_frames:
            image_path = f'{clip_path}/{frame_id}.jpg'
            image_path = os.path.join(image_path)
            image = Image.open(image_path).convert('RGB')

            image = self.transform(image=np.array(image))['image']
            seq_of_frames.append(image)

        seq_of_frames = torch.stack(seq_of_frames)

        return seq_of_frames, labels





def get_B5_loaders(dataset_root, batch_size=32, sequence_length=9):
    """B5: LSTM on player-level"""
    return create_data_loaders(
        dataset_root=dataset_root,
        mode='B5',
        batch_size=batch_size,
        sequence_length=sequence_length
    )


def get_B6_loaders(dataset_root, batch_size=32, sequence_length=9):
    """B6: LSTM on image level only"""
    return create_data_loaders(
        dataset_root=dataset_root,
        mode='B6',
        batch_size=batch_size,
        sequence_length=sequence_length
    )


def get_B7_loaders(dataset_root, batch_size=32, sequence_length=9):
    """B7: Full model V1 - LSTM on both player and frame level"""
    return create_data_loaders(
        dataset_root=dataset_root,
        mode='B7',
        batch_size=batch_size,
        sequence_length=sequence_length
    )


def get_B8_loaders(dataset_root, batch_size=32, sequence_length=9):
    """B8: Team-separated representation"""
    return create_data_loaders(
        dataset_root=dataset_root,
        mode='B8',
        batch_size=batch_size,
        sequence_length=sequence_length,
        team_split=True
    )


if __name__ == "__main__":
    # Example usage
    dataset_root = "path/to/volleyball/dataset"

    # Test B1 loader
    train_loader, val_loader, test_loader = get_B1_loaders(dataset_root)

    for batch_idx, (data, target) in enumerate(train_loader):
        print(f"Batch {batch_idx}: Data shape: {data.shape}, Target shape: {target.shape}")
        break