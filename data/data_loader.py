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


class get_B5_A_loaders(Dataset):
    def __init__(self, videos_path,annot_path, split, transform=None):
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
        # print("Available videos in data:", list(dataset_dict.keys()))
        # print("Available labels in labels_dict:", list(label_dict.keys()))

        self.data=[]
        self.clip={}
        
        with open(annot_path,'rb')as file:
            videos_annot=pickle.load(file)


        for idx in split:
            video_annot=videos_annot[str(idx)]

            for clip in video_annot.keys():
                frames_data = video_annot[str(clip)]['frame_boxes_dct']
                category = video_annot[str(clip)]['category']
                dir_frames = list(video_annot[str(clip)]['frame_boxes_dct'].items()) # 9 frames per clip

                
                self.dict_all_players = {}

                for frame_id,boxes in dir_frames:
                    image_path=f'{videos_path}/{str(idx)}/{str(clip)}/{frame_id}.jpg'
                    image_path=os.path.join(image_path)
                    

                    
                    for box_info in boxes: # 12 players boxes per frame
                        x1,y1,x2,y2=box_info.box
                        if box_info.player_ID not in self.dict_all_players:
                            self.dict_all_players[box_info.player_ID] = []
                        self.dict_all_players[box_info.player_ID].append(box_info)

                        
                self.data.append(
                    {
                        'clip_path':f'{videos_path}/{str(idx)}/{str(clip)}',
                        'dict_all_players':self.dict_all_players,
                        'frame_path':f'{videos_path}/{str(idx)}/{str(clip)}/{frame_id}.jpg',
                        'category':category
                    }
                )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample=self.data[idx]
        num_classes=len(self.categories_dct)
        # labels=torch.zeros(num_classes)

        dict_all_players=sample['dict_all_players']
        clip_path=sample['clip_path']
        print(f"Processing clip: {clip_path}")
        all_players=[]
        all_labels=[]
        for player_id, boxes in dict_all_players.items():
            seq_player=[]
            labels=torch.zeros(num_classes)
            
            for box_info in boxes:
                category = box_info.category
                labels[self.categories_dct[category]] = 1
                x1, y1, x2, y2 = box_info.box
                # Crop the image based on the bounding box
                frame_path = f"{clip_path}/{box_info.frame_ID}.jpg"
                image = Image.open(frame_path).convert('RGB')
                image = image.crop((x1, y1, x2, y2))
                if self.transform:
                    image = self.transform(image=np.array(image))['image']
                seq_player.append(image)
                
            if len(seq_player) < 9:
                seq_player += [torch.zeros(3, 224, 224)] * (9 - len(seq_player))
            # labels[self.categories_dct[category]] = 1
            seq_player = torch.stack(seq_player)
            all_players.append(seq_player)
            all_labels.append(labels.clone())

        if len(all_players) < 12:
            all_players += [torch.zeros(9, 3, 224, 224)] * (12 - len(all_players))
            all_labels += [torch.zeros(len(self.categories_dct))] * (12 - len(all_labels))
        all_players = torch.stack(all_players)
        labels = torch.stack(all_labels)

        print(f"all_players shape: {all_players.shape}, all_labels shape: {labels.shape}")
        # Ensure the shape is (12, 9, 3, 224, 224)
        # if all_players.shape[0] < 12:
        #     all_players = torch.cat([all_players, torch.zeros(12 - all_players.shape[0], 9, 3, 224, 224)], dim=0)
        # if all_players.shape[1] < 9:
        #     all_players = torch.cat([all_players, torch.zeros(all_players.shape[0], 9 - all_players.shape[1], 3, 224, 224)], dim=1)
        # if labels.shape[0] < 12:
        #     labels = torch.cat([labels, torch.zeros(12 - labels.shape[0], len(self.categories_dct))], dim=0)
        # if labels.shape[1] < len(self.categories_dct):
        #     labels = torch.cat([labels, torch.zeros(labels.shape[0], len(self.categories_dct) - labels.shape[1])], dim=1)
        
       
        return all_players, labels

class get_B5_B_loaders(Dataset):
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
        self.clip={}
        
        with open(annot_path,'rb')as file:
            videos_annot=pickle.load(file)


        for idx in split:
            video_annot=videos_annot[str(idx)]

            for clip in video_annot.keys():
                frames_data = video_annot[str(clip)]['frame_boxes_dct']
                category = video_annot[str(clip)]['category']
                dir_frames = list(video_annot[str(clip)]['frame_boxes_dct'].items()) # 9 frames per clip

                
                self.dict_all_players = {}

                for frame_id,boxes in dir_frames:
                    image_path=f'{videos_path}/{str(idx)}/{str(clip)}/{frame_id}.jpg'
                    image_path=os.path.join(image_path)
                    

                    
                    for box_info in boxes: # 12 players boxes per frame
                        x1,y1,x2,y2=box_info.box
                        if box_info.player_ID not in self.dict_all_players:
                            self.dict_all_players[box_info.player_ID] = []
                        self.dict_all_players[box_info.player_ID].append(box_info)

                        
                self.data.append(
                    {
                        'clip_path':f'{videos_path}/{str(idx)}/{str(clip)}',
                        'dict_all_players':self.dict_all_players,
                        'frame_path':f'{videos_path}/{str(idx)}/{str(clip)}/{frame_id}.jpg',
                        'category':category
                    }
                )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample=self.data[idx]
        num_classes=len(self.categories_dct)
        category=sample['category']
        labels=torch.zeros(num_classes)
        labels[self.categories_dct[category]] = 1

        dict_all_players=sample['dict_all_players']
        clip_path=sample['clip_path']

        all_players=[]
        # all_labels=[]
        for player_id, boxes in dict_all_players.items():
            seq_player=[]
            #labels=torch.zeros(num_classes)
            
            for box_info in boxes:
                #category = box_info.category
                #labels[self.categories_dct[category]] = 1
                x1, y1, x2, y2 = box_info.box
                # Crop the image based on the bounding box
                frame_path = f"{clip_path}/{box_info.frame_ID}.jpg"
                image = Image.open(frame_path).convert('RGB')
                image = image.crop((x1, y1, x2, y2))
                if self.transform:
                    image = self.transform(image=np.array(image))['image']
                seq_player.append(image)
                
            if len(seq_player) < 9:
                seq_player += [torch.zeros(3, 224, 224)] * (9 - len(seq_player))
            # labels[self.categories_dct[category]] = 1
            seq_player = torch.stack(seq_player)
            all_players.append(seq_player)

        if len(all_players) < 12:
            all_players += [torch.zeros(9, 3, 224, 224)] * (12 - len(all_players))

        all_players = torch.stack(all_players)
        
       
        return all_players, labels

class get_B6_loaders(Dataset):
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
        self.clip={}
        
        with open(annot_path,'rb')as file:
            videos_annot=pickle.load(file)


        for idx in split:
            video_annot=videos_annot[str(idx)]

            for clip in video_annot.keys():

                category = video_annot[str(clip)]['category']
                dir_frames = list(video_annot[str(clip)]['frame_boxes_dct'].items()) # 9 frames per clip

                
                frame_data = []

                for frame_id,boxes in dir_frames:

                    frame_path=f'{videos_path}/{str(idx)}/{str(clip)}/{frame_id}.jpg'
                    
                    frame_boxes=[]
                    
                    for box_info in boxes: 

                        frame_boxes.append(box_info)
                        
                    frame_data.append((frame_path,frame_boxes))
                        
                self.data.append(
                    {
                        'frame_data':frame_data,
                        'category':category
                    }
                )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample=self.data[idx]
        num_classes=len(self.categories_dct)
        category=sample['category']
        label=torch.zeros(num_classes)
        label[self.categories_dct[category]] = 1

        frame_data=sample['frame_data']

        clip=[]
        labels=[]
        for frame_path, frame_boxes in frame_data:
            seq_player=[]
            #labels=torch.zeros(num_classes)

            frame = cv2.imread(frame_path)
            
            for box_info in frame_boxes:

                x1, y1, x2, y2 = box_info.box

                # Crop the image based on the bounding box
                person_crop = frame[y1:y2, x1:x2]
                #image = image.crop((x1, y1, x2, y2))

                if self.transform:
                    transformed = self.transform(image=person_crop)
                    image = transformed['image']
                seq_player.append(image)

            seq_player = torch.stack(seq_player)
            clip.append(seq_player)
            labels.append(label)

        clip = torch.stack(clip).permute(1, 0, 2, 3, 4)
        labels = torch.stack(labels)

        return clip, labels


# get_B7_step_A
class PersonActivityTempDataset(Dataset):
    def __init__(self, videos_path,annot_path, split, transform=None):
        self.samples = []
        self.videos_path=videos_path
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
        
        with open(annot_path,'rb')as file:
            videos_annot=pickle.load(file)

        self.frames_index=[]

        for idx in split:
            video_annot=videos_annot[str(idx)]

            for clip in video_annot.keys():

                #category = video_annot[str(clip)]['category']
                dir_frames = list(video_annot[str(clip)]['frame_boxes_dct'].items()) # 9 frames per clip

                
                frame_seq = []

                for frame_id,boxes in dir_frames:

                    frame_seq.append({
                        'frame_id':frame_id,
                        'boxes':boxes
                    })
                        
                    #frame_data.append((frame_path,frame_boxes))
                        
                self.frames_index.append(
                    {
                        'frame_seq':frame_seq,
                        'idx':idx,
                        'clip':clip
                    }
                )

    def __len__(self):
        return len(self.frames_index)

    

    def __getitem__(self, idx):
        sample=self.frames_index[idx]
        num_classes=len(self.categories_dct)
        #category=sample['category']
        # label=torch.zeros(num_classes)
        # label[self.categories_dct[category]] = 1

        seq_crops=[]
        seq_labels=[]

        frame_seq=sample['frame_seq']

        for frame_data in frame_seq:
            # print("idx",sample["idx"])
            # print("clip",sample['clip'])
            idx=str(sample["idx"])
            clip=str(sample['clip'])
            # print("idx2",idx)
            # print("clip2",clip)
            frame_path = f"{self.videos_path}/{idx}/{clip}/{frame_data['frame_id']}.jpg"
            # print(frame_path)
            frame = cv2.imread(frame_path)

            frame_crops=[]
            frame_lables=[]
            for box in frame_data['boxes']:
                seq_player=[]
                #labels=torch.zeros(num_classes)
                
                x1, y1, x2, y2 = box.box

                # Crop the image based on the bounding box
                person_crop = frame[y1:y2, x1:x2]
                #image = image.crop((x1, y1, x2, y2))

                if self.transform:
                    transformed = self.transform(image=person_crop)
                    person_crop = transformed['image']

                label = np.zeros(len(self.categories_dct))
                label[self.categories_dct[box.category]] = 1

                frame_crops.append(person_crop)
                frame_lables.append(label)

            seq_crops.append(np.stack(frame_crops))
            seq_labels.append(np.stack(frame_lables))


        # Stack and transpose to get (num_people, num_frames, C, H, W)

        seq_crops = np.stack(seq_crops)
        seq_crops = np.transpose(seq_crops, (1, 0, 2, 3, 4))

        seq_labels = np.stack(seq_labels)
        seq_labels = np.transpose(seq_labels, (1, 0, 2))

        return torch.from_numpy(seq_crops), torch.from_numpy(seq_labels)


# get_B7_step_B and get_B8
class GroupActivityTempDataset(Dataset):
    def __init__(self, videos_path,annot_path, split, sort = False, transform=None):
        self.samples = []
        self.transform = transform
        self.sort = sort # If True, prepares data for the 2-group model
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
        self.clip={}
        
        with open(annot_path,'rb')as file:
            videos_annot=pickle.load(file)


        for idx in split:
            video_annot=videos_annot[str(idx)]

            for clip in video_annot.keys():

                category = video_annot[str(clip)]['category']
                dir_frames = list(video_annot[str(clip)]['frame_boxes_dct'].items()) # 9 frames per clip

                
                frame_data = []

                for frame_id,boxes in dir_frames:

                    frame_path=f'{videos_path}/{str(idx)}/{str(clip)}/{frame_id}.jpg'
                    
                    frame_boxes=[]
                    
                    for box_info in boxes: 

                        frame_boxes.append(box_info)
                        
                    frame_data.append((frame_path,frame_boxes))
                        
                self.data.append(
                    {
                        'frame_data':frame_data,
                        'category':category
                    }
                )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample=self.data[idx]
        num_classes=len(self.categories_dct)
        category=sample['category']
        label=torch.zeros(num_classes)
        label[self.categories_dct[category]] = 1

        frame_data=sample['frame_data']

        clip=[]
        labels=[]
        for frame_path, frame_boxes in frame_data:
            seq_player=[]
            orders=[]
            #labels=torch.zeros(num_classes)

            frame = cv2.imread(frame_path)
            
            for box_info in frame_boxes:

                x1, y1, x2, y2 = box_info.box

                x_center = (x1 + x2) // 2
                orders.append(x_center)

                # Crop the image based on the bounding box
                person_crop = frame[y1:y2, x1:x2]

                if self.transform:
                    transformed = self.transform(image=person_crop)
                    image = transformed['image']
                seq_player.append(image)

            if self.sort:
                # Sort seq_player based on orders
                orders_with_images = list(zip(orders, seq_player))
                orders_with_images.sort(key=lambda x: x[0])  # Sort by x_center
                seq_player = [img for _, img in orders_with_images]

            seq_player = torch.stack(seq_player)
            clip.append(seq_player)
            labels.append(label)

        clip = torch.stack(clip).permute(1, 0, 2, 3, 4)
        labels = torch.stack(labels)

        return clip, labels


