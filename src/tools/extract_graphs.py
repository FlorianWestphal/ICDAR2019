import util.graph_extractor as ge
from PIL import Image
import os
import networkx as nx


IMG_SHAPE = (280, 280)


def main(src_path, dest_path, d_v=40, d_h=40):

    extractor = ge.GraphExtractor(d_v=d_v, d_h=d_h)
    for i in range(10):
        s_p = os.path.join(src_path, str(i))
        d_p = os.path.join(dest_path, str(i))
        os.mkdir(d_p)
        files = os.listdir(s_p)
        for f in files:
            img = Image.open(os.path.join(s_p, f))
            img = img.resize(IMG_SHAPE)
            g = extractor.extract_graph(img)
            name = '{}.gpickle'.format(f.split('.')[0])
            nx.write_gpickle(g, os.path.join(d_p, name))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
                        description='Extract graphs from all images of the '
                        'given source folder structure and store the graphs '
                        'in the corresponding destination folder structure '
                        'using projection profiles.',
                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('src_path', help='Path to source folder structure.')
    parser.add_argument('dest_path', help='Path to destination folder '
                        'structure')
    parser.add_argument('--d_v', help='Vertical distance.',
                        type=int, default=40)
    parser.add_argument('--d_h', help='Horizontal distance.',
                        type=int, default=40)

    args = vars(parser.parse_args())
    main(**args)
