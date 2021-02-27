import json
import imageio
from tqdm.auto import tqdm
import time
from PIL import Image, ImageFont, ImageDraw
import re
import random
from matplotlib import pyplot as plt
import os
import string
from IPython.display import FileLink, HTML, display

def random_string_id(L=6):
    """Random ID."""
    char_set = string.ascii_uppercase + string.digits
    return ''.join(random.sample(char_set * L, L))

def display_image(fn):
    """Show image in Jupyter."""
    fn = os.path.relpath(fn)
    link = FileLink(fn)
    img = HTML('<img src="{}">'.format(fn))
    return display(link, img)

def get_trial_by_epochs(results):
    """Get the dict epoch -> metrics."""
    trial = []
    with open(results) as f:
        for line in tqdm(f.readlines()):
            line = line.strip()
            data = json.loads(line)
            trial.append(data)

    trial_by_epochs = {}
    for data in trial:
        if 'epochs' in data:
            trial_by_epochs[data['epochs']] = data
    return trial_by_epochs

def img_extend(img, add_w=0, add_h=0):
    """Add pixels below and on the right."""
    img_w, img_h = img.size
    background = Image.new('RGBA', (img_w + add_w, img_h + add_h), (255, 255, 255, 255))
    background.paste(img, (0, 0))
    return background

def image_add_text(input_filename, text, color=(0, 0, 0, 50),
                   offset=30,
                   output_filename=None):
    """Add text to the image."""
    if output_filename is None:
        ext = input_filename.split('.')[-1]
        output_filename = f"{input_filename}_withtext.{ext}"
    
    if not os.path.isfile(input_filename):
        return None
    
    img = Image.open(input_filename)
    img = img_extend(img, add_h=offset)
    font = ImageFont.truetype('/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf', 20)
    img_edit = ImageDraw.Draw(img)
    img_edit.text((0, img.height - offset), text, color, font=font)
    img.save(output_filename)
    return output_filename

def progress(value, total, width, symbol='#', empty=' '):
    """Get a string representing the progress."""
    frac = 1. * value / total
    pixels = round(frac * width)
    left_pixels = width - pixels
    return '[' + (symbol * pixels) + (empty * left_pixels) + ']'

def images_to_gif(files_withtext, base_dir, key, duration=6):
    """Convert a list of filenames into one .gif."""
    dirname = os.path.abspath(os.path.join(base_dir, '..'))
    basename = key
    images = []
    for filename in tqdm(files_withtext):
        if filename is None:
            continue
        img_pil = imageio.imread(filename)
        images.append(img_pil)
        
    out_fn = os.path.join(dirname, basename + '_all.gif')
    
    frame_duration = duration / len(images)
    imageio.mimsave(out_fn, images, format='GIF', duration=frame_duration)
    return out_fn

def add_text_batch(base_dir, trial_by_epochs, fn='CausalModel.png'):
    """Add epoch info to all images, return the new filenames."""
    epochs = os.listdir(base_dir)
    epochs = sorted([x for x in epochs if x.startswith('epoch')])
    files = [os.path.join(base_dir, epoch, fn) for epoch in epochs]
    out_files = []
    for i, (f, epoch) in tqdm(enumerate(zip(files, epochs))):
        epoch = int(epoch[len('epoch'):])
        info = trial_by_epochs[epoch]
        time_hms = time.strftime('%H:%M:%S', time.gmtime(info['time_total_s']))
        progress_val = progress(i, total=len(files), width=10)
        txt = "Epoch %05d time %s %s" % (epoch, time_hms, progress_val)
        out_file = image_add_text(f, txt)
        out_files.append(out_file)
    return out_files

def gifify_trial_images(trial_by_epochs, filename):
    """Get a .gif file from the trial images."""
    base_dir = trial_by_epochs[min(trial_by_epochs.keys())]['config']['base_dir']
    files_withtext = add_text_batch(base_dir, trial_by_epochs, filename)
    out_gif = images_to_gif(files_withtext, base_dir, filename)
    return out_gif

def plot_hist_values(trial_by_epochs, key, n_images=50, log=True):
    """Plot cumulative charts for a key in data."""
    
    epochs_all = sorted(trial_by_epochs.keys())
    
    epochs_present = os.listdir(trial_by_epochs[0]['config']['base_dir'])
    epochs_present = [x for x in epochs_present if x.startswith('epoch')]
    epochs_present = [int(x[len('epoch'):]) for x in epochs_present]
    epochs_present = sorted(epochs_present)
    
    key_alphanum = re.sub('[^0-9a-zA-Z]+', '', key)
    img_fn = key_alphanum + '.png'
    skip = len(epochs_present) // n_images

    values = []
    fns = []
    epochs_used = []
    for epoch in epochs_present:
        out_dir = os.path.join(trial_by_epochs[0]['config']['base_dir'], 'epoch%05d' % epoch)
        fn = os.path.join(out_dir, img_fn)
        try:
            os.unlink(fn)
        except FileNotFoundError:
            pass
        
    epoch_data_lastadd = -1

    for epoch in tqdm(epochs_present[::skip]):
        out_dir = os.path.join(trial_by_epochs[0]['config']['base_dir'], 'epoch%05d' % epoch)
        fn = os.path.join(out_dir, img_fn)
        if not os.path.isdir(out_dir):
            continue

        for epoch_add in epochs_all:
            if epoch_add <= epoch and epoch_add > epoch_data_lastadd:
                epochs_used.append(epoch_add)
                values.append(trial_by_epochs[epoch_add][key])
                epoch_data_lastadd = epoch_add
        
        f = plt.figure(figsize=(8, 5))
        plt.title(key)
        plt.plot(epochs_used, values)
        if log:
            plt.yscale('log')
        plt.savefig(fn, bbox_inches='tight')
        fns.append(fn)
        f.clear()
        plt.close(f)
        
    return img_fn