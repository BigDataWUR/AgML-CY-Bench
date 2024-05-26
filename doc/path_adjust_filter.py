import pandocfilters as pf

def adjust_paths(key, value, format, meta):
    if key == 'Link':
        [[ident, classes, kvs], txt, target] = value
        href, title = target
        # Adjust the href as needed
        prepend_path = 'http://github.com/BigDataWUR/AgML-crop-yield-forecasting/blob/main'
        if not href.startswith(('http://', 'https://', '#')):  # Avoid modifying absolute URLs and anchor links
            href = prepend_path + href
        return pf.Link([ident, classes, kvs], txt, [href, title])

if __name__ == "__main__":
    import sys
    pf.toJSONFilter(adjust_paths)