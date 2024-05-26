import pandocfilters as pf

def adjust_paths(key, value, format, meta):
    if key == 'Link':
        prepend_path = 'https://github.com/BigDataWUR/AgML-crop-yield-forecasting/tree/main/'
        [[ident, classes, kvs], txt, target] = value
        href, title = target
        # Adjust the href as needed
        if not href.startswith(('http://', 'https://', '#')):  # Avoid modifying absolute URLs and anchor links
            href = prepend_path + href
        return pf.Link([ident, classes, kvs], txt, [href, title])
    if key == 'Image':
        [[ident, classes, kvs], alt, [src, title]] = value
        prepend_path = 'https://raw.githubusercontent.com/BigDataWUR/AgML-crop-yield-forecasting/main/'
        if not src.startswith(('http://', 'https://')):  # Avoid modifying absolute URLs
            src = prepend_path + src
        return pf.Image([ident, classes, kvs], alt, [src, title])

if __name__ == "__main__":
    import sys
    pf.toJSONFilter(adjust_paths)