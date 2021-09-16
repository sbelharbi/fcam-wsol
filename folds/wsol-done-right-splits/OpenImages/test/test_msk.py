def get_mask_paths(localization):
    """
    localization.txt (for masks) has the structure

    <path>,<link_to_mask_file>,<link_to_ignore_mask_file>
    path/to/image1.jpg,path/to/mask1a.png,path/to/ignore1.png
    path/to/image1.jpg,path/to/mask1b.png,
    path/to/image2.jpg,path/to/mask2a.png,path/to/ignore2.png
    path/to/image3.jpg,path/to/mask3a.png,path/to/ignore3.png
    ...

    One image may contain multiple masks (multiple mask paths for same image).
    One image contains only one ignore mask.
    """
    mask_paths = {}
    ignore_paths = {}
    with open(localization) as f:
        for line in f.readlines():
            image_id, mask_path, ignore_path = line.strip('\n').split(',')
            if image_id in mask_paths:
                mask_paths[image_id].append(mask_path)
                assert (len(ignore_path) == 0)
            else:
                mask_paths[image_id] = [mask_path]
                ignore_paths[image_id] = ignore_path
    return mask_paths, ignore_paths



mask_paths, ignore_paths = get_mask_paths(localization='localization.txt')
for i in ignore_paths:
    if ignore_paths[i] == '':
        print(i)
