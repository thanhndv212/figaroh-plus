# figaroh Documentation

## Building and Viewing Documentation

1. Install Sphinx and theme:
```bash
pip install sphinx sphinx-rtd-theme
```

2. Build HTML docs:
```bash
cd docs
make html
```

3. View documentation locally using one of these methods:

a. Using Python's built-in HTTP server:
```bash
cd build/html
python -m http.server 8000
```
Then open http://localhost:8000 in your browser

b. Using PHP's built-in server:
```bash 
cd build/html
php -S localhost:8000
```

c. Direct file access:
- Open `build/html/index.html` in your web browser
- Navigate using the sidebar menu

## Development

To auto-rebuild documentation when files change:

```bash
sphinx-autobuild source build/html
```

This will start a server at http://localhost:8000 and rebuild docs when source files change.

## Deployment 

To deploy to GitHub Pages:

1. Build documentation:
```bash
make html
```

2. Copy contents of `build/html` to your gh-pages branch
```bash
cp -r build/html/* /path/to/gh-pages/
```

3. Push to GitHub:
```bash 
cd /path/to/gh-pages
git add .
git commit -m "Update documentation"
git push origin gh-pages
```
