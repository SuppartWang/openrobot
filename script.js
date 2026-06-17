// OpenRobot Landing Page Scripts

document.addEventListener('DOMContentLoaded', function() {
    // 生成 Hero 粒子背景
    createParticles();
    
    // 滚动动画
    initScrollAnimations();
    
    // 导航栏滚动效果
    initNavbarScroll();
});

function createParticles() {
    const container = document.getElementById('particles');
    if (!container) return;
    
    const particleCount = 50;
    
    for (let i = 0; i < particleCount; i++) {
        const particle = document.createElement('div');
        particle.className = 'particle';
        
        // 随机位置和动画参数
        const left = Math.random() * 100;
        const delay = Math.random() * 15;
        const duration = 10 + Math.random() * 20;
        const size = 2 + Math.random() * 4;
        
        particle.style.left = `${left}%`;
        particle.style.animationDelay = `${delay}s`;
        particle.style.animationDuration = `${duration}s`;
        particle.style.width = `${size}px`;
        particle.style.height = `${size}px`;
        particle.style.opacity = 0.2 + Math.random() * 0.5;
        
        container.appendChild(particle);
    }
}

function initScrollAnimations() {
    const observerOptions = {
        root: null,
        rootMargin: '0px 0px -50px 0px',
        threshold: 0.05
    };
    
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('is-visible');
                observer.unobserve(entry.target);
            }
        });
    }, observerOptions);
    
    // 为所有卡片和层添加观察
    const animatedElements = document.querySelectorAll(
        '.principle-card, .arch-layer, #layers .group, #flow ol li'
    );
    
    animatedElements.forEach((el) => {
        el.classList.add('will-animate');
        observer.observe(el);
    });
}

function initNavbarScroll() {
    const nav = document.querySelector('nav');
    
    window.addEventListener('scroll', () => {
        const currentScroll = window.pageYOffset;
        
        if (currentScroll > 50) {
            nav.classList.add('shadow-lg', 'shadow-slate-900/50');
            nav.style.background = 'rgba(2, 6, 23, 0.95)';
        } else {
            nav.classList.remove('shadow-lg', 'shadow-slate-900/50');
            nav.style.background = 'rgba(2, 6, 23, 0.8)';
        }
    });
}

// 平滑滚动（对所有锚点链接）
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function(e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));
        if (target) {
            target.scrollIntoView({
                behavior: 'smooth',
                block: 'start'
            });
        }
    });
});
