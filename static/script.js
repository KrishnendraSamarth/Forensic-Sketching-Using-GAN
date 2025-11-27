/**
 * Forensic Sketching using GAN - Client-side JavaScript
 * Handles animations, interactions, form submission, and image generation
 */

document.addEventListener('DOMContentLoaded', function() {
    // Initialize particle animation
    initParticles();
    
    // Initialize smooth scroll
    initSmoothScroll();
    
    // Initialize navbar scroll effect
    initNavbarScroll();
    
    // Initialize fade-in on scroll
    initFadeInOnScroll();

    // Form elements
    const form = document.getElementById('uploadForm');
    const sketchInput = document.getElementById('sketchInput');
    const uploadArea = document.getElementById('uploadArea');
    const uploadLabel = document.getElementById('uploadLabel');
    const uploadText = document.getElementById('uploadText');
    const fileInfo = document.getElementById('fileInfo');
    const previewContainer = document.getElementById('previewContainer');
    const previewImage = document.getElementById('previewImage');
    const removeBtn = document.getElementById('removeBtn');
    const generateBtn = document.getElementById('generateBtn');
    const resultSection = document.getElementById('resultSection');
    const sketchPreview = document.getElementById('sketchPreview');
    const generatedImage = document.getElementById('generatedImage');
    const downloadBtn = document.getElementById('downloadBtn');
    const loading = document.getElementById('loading');
    const error = document.getElementById('error');
    const errorMessage = document.getElementById('errorMessage');
    let currentImageUrl = null;
    let currentGeneratedUrl = null;

    /**
     * Initialize particle animation in hero section
     */
    function initParticles() {
        const particlesContainer = document.getElementById('particles');
        if (!particlesContainer) return;

        const particleCount = 50;
        for (let i = 0; i < particleCount; i++) {
            const particle = document.createElement('div');
            particle.className = 'particle';
            particle.style.left = Math.random() * 100 + '%';
            particle.style.top = Math.random() * 100 + '%';
            particle.style.animationDelay = Math.random() * 15 + 's';
            particle.style.animationDuration = (10 + Math.random() * 10) + 's';
            particlesContainer.appendChild(particle);
        }
    }

    /**
     * Initialize smooth scroll for anchor links
     */
    function initSmoothScroll() {
        // Smooth scroll for "Try Sketch Generator" button
        const trySketchBtn = document.getElementById('trySketchBtn');
        if (trySketchBtn) {
            trySketchBtn.addEventListener('click', function(e) {
                e.preventDefault();
                const tryItSection = document.getElementById('try-it');
                if (tryItSection) {
                    tryItSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
                }
            });
        }

        // Smooth scroll for nav links
        document.querySelectorAll('a[href^="#"]').forEach(anchor => {
            anchor.addEventListener('click', function(e) {
                const href = this.getAttribute('href');
                if (href === '#' || href === '#home') return;
                
                e.preventDefault();
                const target = document.querySelector(href);
                if (target) {
                    const offset = 80; // Navbar height
                    const targetPosition = target.offsetTop - offset;
                    window.scrollTo({
                        top: targetPosition,
                        behavior: 'smooth'
                    });
                }
            });
        });
    }

    /**
     * Initialize navbar scroll effect
     */
    function initNavbarScroll() {
        const navbar = document.querySelector('.navbar');
        if (!navbar) return;

        window.addEventListener('scroll', function() {
            if (window.scrollY > 50) {
                navbar.classList.add('scrolled');
            } else {
                navbar.classList.remove('scrolled');
            }
        });
    }

    /**
     * Initialize fade-in on scroll animation
     */
    function initFadeInOnScroll() {
        const fadeElements = document.querySelectorAll('.fade-in');
        if (fadeElements.length === 0) return;

        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    entry.target.classList.add('visible');
                }
            });
        }, {
            threshold: 0.1,
            rootMargin: '0px 0px -50px 0px'
        });

        fadeElements.forEach(element => {
            observer.observe(element);
        });
    }

    /**
     * Display preview of uploaded image
     */
    function showPreview(file) {
        if (!file) return;

        const reader = new FileReader();
        reader.onload = function(e) {
            // Clean up previous image URL
            if (currentImageUrl) {
                URL.revokeObjectURL(currentImageUrl);
            }

            currentImageUrl = e.target.result;
            previewImage.src = currentImageUrl;
            if (sketchPreview) sketchPreview.src = currentImageUrl;
            
            // Show preview, hide upload label
            previewContainer.style.display = 'block';
            uploadLabel.style.display = 'none';
            uploadArea.classList.add('has-image');
            
            // Enable generate button
            if (generateBtn) generateBtn.disabled = false;
        };
        reader.readAsDataURL(file);
    }

    // Handle file input change
    if (sketchInput) {
        sketchInput.addEventListener('change', function(e) {
            const file = e.target.files[0];
            if (file) {
                showPreview(file);
            }
        });
    }

    // Remove preview
    if (removeBtn) {
        removeBtn.addEventListener('click', function(e) {
            e.stopPropagation();
            resetUpload();
        });
    }

    function resetUpload() {
        if (sketchInput) sketchInput.value = '';
        if (previewContainer) previewContainer.style.display = 'none';
        if (uploadLabel) uploadLabel.style.display = 'flex';
        if (uploadArea) {
            uploadArea.classList.remove('has-image', 'drag-over');
        }
        if (generateBtn) generateBtn.disabled = true;
        hideResults();
        hideError();
        
        if (currentImageUrl) {
            URL.revokeObjectURL(currentImageUrl);
            currentImageUrl = null;
        }
    }

    // Drag and drop functionality
    if (uploadArea) {
        uploadArea.addEventListener('dragover', function(e) {
            e.preventDefault();
            uploadArea.classList.add('drag-over');
        });

        uploadArea.addEventListener('dragleave', function(e) {
            e.preventDefault();
            uploadArea.classList.remove('drag-over');
        });

        uploadArea.addEventListener('drop', function(e) {
            e.preventDefault();
            uploadArea.classList.remove('drag-over');

            const file = e.dataTransfer.files[0];
            if (file && file.type.startsWith('image/')) {
                try {
                    const dataTransfer = new DataTransfer();
                    dataTransfer.items.add(file);
                    if (sketchInput) sketchInput.files = dataTransfer.files;
                } catch (err) {
                    console.warn('DataTransfer not supported, using fallback');
                }
                showPreview(file);
            } else {
                showError('Please drop a valid image file');
            }
        });
    }

    // Handle form submission
    if (form) {
        form.addEventListener('submit', async function(e) {
            e.preventDefault();

            // Hide previous results and errors
            hideResults();
            hideError();

            // Check if file is selected
            if (!sketchInput || !sketchInput.files || sketchInput.files.length === 0) {
                showError('Please select an image file');
                return;
            }

            const file = sketchInput.files[0];

            // Validate file type
            if (!file.type.startsWith('image/')) {
                showError('Please select a valid image file');
                return;
            }

            // Validate file size (16MB)
            if (file.size > 16 * 1024 * 1024) {
                showError('File size must be less than 16MB');
                return;
            }

            // Show loading state
            showLoading();
            if (generateBtn) generateBtn.disabled = true;

            try {
                // Create FormData
                const formData = new FormData();
                formData.append('sketch', file);

                // Send POST request
                const response = await fetch('/generate', {
                    method: 'POST',
                    body: formData
                });

                // Check if response is ok
                if (!response.ok) {
                    const errorData = await response.json().catch(() => ({ error: 'Unknown error occurred' }));
                    throw new Error(errorData.error || `Server error: ${response.status}`);
                }

                // Get image blob
                const imageBlob = await response.blob();
                
                // Clean up previous generated image URL
                if (currentGeneratedUrl) {
                    URL.revokeObjectURL(currentGeneratedUrl);
                }
                
                currentGeneratedUrl = URL.createObjectURL(imageBlob);

                // Display generated image
                if (generatedImage) generatedImage.src = currentGeneratedUrl;
                
                // Set sketch preview (use the uploaded image)
                if (currentImageUrl && sketchPreview) {
                    sketchPreview.src = currentImageUrl;
                }
                
                if (resultSection) resultSection.style.display = 'block';

                // Hide loading
                hideLoading();

                // Scroll to results smoothly
                if (resultSection) {
                    resultSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
                }

            } catch (err) {
                console.error('Error:', err);
                showError(err.message || 'Failed to generate face. Please try again.');
                hideLoading();
            } finally {
                if (generateBtn) generateBtn.disabled = false;
            }
        });
    }

    function showLoading() {
        if (loading) loading.style.display = 'block';
        if (resultSection) resultSection.style.display = 'none';
    }

    function hideLoading() {
        if (loading) loading.style.display = 'none';
    }

    function showError(message) {
        if (errorMessage) errorMessage.textContent = message;
        if (error) error.style.display = 'block';
    }

    function hideError() {
        if (error) error.style.display = 'none';
    }

    function hideResults() {
        if (resultSection) resultSection.style.display = 'none';
    }

    // Download button handler
    if (downloadBtn) {
        downloadBtn.addEventListener('click', function() {
            if (currentGeneratedUrl) {
                const link = document.createElement('a');
                link.href = currentGeneratedUrl;
                link.download = 'generated_face.png';
                document.body.appendChild(link);
                link.click();
                document.body.removeChild(link);
            }
        });
    }

    // Clean up object URLs when page unloads
    window.addEventListener('beforeunload', function() {
        if (currentImageUrl) {
            URL.revokeObjectURL(currentImageUrl);
        }
        if (currentGeneratedUrl) {
            URL.revokeObjectURL(currentGeneratedUrl);
        }
    });

    // Initialize: disable generate button if no file selected
    if (generateBtn) generateBtn.disabled = true;
    
});
