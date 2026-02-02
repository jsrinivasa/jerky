#!/usr/bin/env python3

import h5py
import sys

def inspect_hdf5(filepath):
    print(f"Inspecting: {filepath}\n")
    
    with h5py.File(filepath, 'r') as f:
        # Attributes
        print("=" * 60)
        print("FILE ATTRIBUTES")
        print("=" * 60)
        for key, value in f.attrs.items():
            print(f"  {key}: {value}")
        
        # Main datasets
        print("\n" + "=" * 60)
        print("ROOT LEVEL DATASETS")
        print("=" * 60)
        for key in f.keys():
            if isinstance(f[key], h5py.Dataset):
                print(f"  /{key}")
                print(f"    Shape: {f[key].shape}")
                print(f"    Dtype: {f[key].dtype}")
        
        # Observations
        if 'observations' in f:
            print("\n" + "=" * 60)
            print("OBSERVATIONS")
            print("=" * 60)
            obs = f['observations']
            for key in obs.keys():
                if isinstance(obs[key], h5py.Dataset):
                    print(f"  /observations/{key}")
                    print(f"    Shape: {obs[key].shape}")
                    print(f"    Dtype: {obs[key].dtype}")
                elif isinstance(obs[key], h5py.Group):
                    print(f"  /observations/{key}/ (group)")
            
            # Images
            if 'images' in obs:
                print("\n" + "=" * 60)
                print("CAMERA IMAGES")
                print("=" * 60)
                images = obs['images']
                for cam_name in images.keys():
                    print(f"  /observations/images/{cam_name}")
                    print(f"    Shape: {images[cam_name].shape}")
                    print(f"    Dtype: {images[cam_name].dtype}")
        
        # Show sample data sizes
        print("\n" + "=" * 60)
        print("EPISODE SUMMARY")
        print("=" * 60)
        if 'action' in f:
            print(f"  Total timesteps: {f['action'].shape[0]}")
            print(f"  Duration: ~{f['action'].shape[0] / 50:.1f} seconds (at 50 FPS)")
        if 'observations/qpos' in f:
            print(f"  DOF: {f['observations/qpos'].shape[1]}")

if __name__ == '__main__':
    if len(sys.argv) > 1:
        inspect_hdf5(sys.argv[1])
    else:
        inspect_hdf5('/home/aloha/aloha_data/aloha_mobile_ctrl_place/episode_0.hdf5')
