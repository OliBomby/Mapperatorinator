/**
 * Mapperatorinator Internationalization (i18n) Module
 * 
 * This module provides internationalization support for the Mapperatorinator UI.
 * It loads language files from /static/i18n/ and applies translations to the DOM.
 * 
 * Usage:
 *   1. Include this script in your HTML: <script src="/static/i18n.js"></script>
 *   2. Call I18n.init() after DOM is ready
 *   3. Add data-i18n attributes to elements that need translation
 *   4. Use I18n.t('key.path') to get translated strings in JavaScript
 */

const I18n = (function() {
    'use strict';

    // Private variables
    let currentLanguage = 'en';
    let translations = {};
    let fallbackTranslations = {};
    const STORAGE_KEY = 'mapperatorinator_language';
    const DEFAULT_LANGUAGE = 'en';
    const SUPPORTED_LANGUAGES = ['en', 'zh-CN'];

    /**
     * Initialize the i18n module
     * @param {string} [lang] - Optional language code to use
     * @returns {Promise} - Resolves when initialization is complete
     */
    async function init(lang) {
        // Determine which language to use
        const savedLang = localStorage.getItem(STORAGE_KEY);
        const browserLang = getBrowserLanguage();
        currentLanguage = lang || savedLang || browserLang || DEFAULT_LANGUAGE;

        // Ensure the language is supported
        if (!SUPPORTED_LANGUAGES.includes(currentLanguage)) {
            currentLanguage = DEFAULT_LANGUAGE;
        }

        try {
            // Always load fallback (English) first
            fallbackTranslations = await loadLanguageFile(DEFAULT_LANGUAGE);

            // Load the target language if different from fallback
            if (currentLanguage !== DEFAULT_LANGUAGE) {
                translations = await loadLanguageFile(currentLanguage);
            } else {
                translations = fallbackTranslations;
            }

            // Apply translations to the DOM
            applyTranslations();

            // Save the language preference
            localStorage.setItem(STORAGE_KEY, currentLanguage);

            console.log(`[i18n] Initialized with language: ${currentLanguage}`);
            return true;
        } catch (error) {
            console.error('[i18n] Failed to initialize:', error);
            translations = fallbackTranslations;
            return false;
        }
    }

    /**
     * Get browser's preferred language
     * @returns {string} - Language code
     */
    function getBrowserLanguage() {
        const browserLang = navigator.language || navigator.userLanguage;
        if (browserLang.startsWith('zh')) {
            return 'zh-CN';
        }
        return browserLang.split('-')[0];
    }

    /**
     * Load a language file
     * @param {string} lang - Language code
     * @returns {Promise<Object>} - Language translations object
     */
    async function loadLanguageFile(lang) {
        const url = `/static/i18n/${lang}.json`;
        const response = await fetch(url);
        if (!response.ok) {
            throw new Error(`Failed to load language file: ${lang}`);
        }
        return await response.json();
    }

    /**
     * Get a translation by key path
     * @param {string} keyPath - Dot-separated path to the translation (e.g., 'buttons.browse')
     * @param {Object} [params] - Optional parameters for string interpolation
     * @returns {string} - Translated string or the key if not found
     */
    function t(keyPath, params) {
        let result = getNestedValue(translations, keyPath);
        
        // Fallback to English if not found
        if (result === undefined) {
            result = getNestedValue(fallbackTranslations, keyPath);
        }

        // Return key if still not found
        if (result === undefined) {
            console.warn(`[i18n] Missing translation: ${keyPath}`);
            return keyPath;
        }

        // Handle string interpolation
        if (params && typeof result === 'string') {
            Object.keys(params).forEach(key => {
                result = result.replace(new RegExp(`\\{${key}\\}`, 'g'), params[key]);
            });
        }

        return result;
    }

    /**
     * Get nested value from object using dot notation
     * @param {Object} obj - Object to search
     * @param {string} path - Dot-separated path
     * @returns {*} - Value at path or undefined
     */
    function getNestedValue(obj, path) {
        if (!obj || !path) return undefined;
        
        const keys = path.split('.');
        let current = obj;
        
        for (const key of keys) {
            if (current === undefined || current === null) {
                return undefined;
            }
            current = current[key];
        }
        
        return current;
    }

    /**
     * Apply translations to all elements with data-i18n attributes
     */
    function applyTranslations() {
        // Update page title
        const pageTitle = t('page.title');
        if (pageTitle && pageTitle !== 'page.title') {
            document.title = pageTitle;
        }

        // Update elements with data-i18n attribute
        document.querySelectorAll('[data-i18n]').forEach(element => {
            const key = element.getAttribute('data-i18n');
            const translation = t(key);
            if (translation !== key) {
                // Check for suffix attribute (used for negative descriptors)
                const suffix = element.getAttribute('data-i18n-suffix');
                element.textContent = suffix ? translation + suffix : translation;
            }
        });

        // Update elements with data-i18n-title attribute (for tooltips)
        document.querySelectorAll('[data-i18n-title]').forEach(element => {
            const key = element.getAttribute('data-i18n-title');
            const translation = t(key);
            if (translation !== key) {
                element.setAttribute('title', translation);
            }
        });

        // Update elements with data-i18n-placeholder attribute
        document.querySelectorAll('[data-i18n-placeholder]').forEach(element => {
            const key = element.getAttribute('data-i18n-placeholder');
            const translation = t(key);
            if (translation !== key) {
                element.setAttribute('placeholder', translation);
            }
        });

        // Update HTML lang attribute
        document.documentElement.lang = currentLanguage;
    }

    /**
     * Change the current language
     * @param {string} lang - Language code
     * @returns {Promise<boolean>} - Resolves to true if successful
     */
    async function setLanguage(lang) {
        if (!SUPPORTED_LANGUAGES.includes(lang)) {
            console.error(`[i18n] Unsupported language: ${lang}`);
            return false;
        }

        if (lang === currentLanguage) {
            return true;
        }

        try {
            if (lang === DEFAULT_LANGUAGE) {
                translations = fallbackTranslations;
            } else {
                translations = await loadLanguageFile(lang);
            }
            
            currentLanguage = lang;
            localStorage.setItem(STORAGE_KEY, lang);
            applyTranslations();
            
            // Trigger custom event for components that need to update
            window.dispatchEvent(new CustomEvent('languageChanged', { detail: { language: lang } }));
            
            console.log(`[i18n] Language changed to: ${lang}`);
            return true;
        } catch (error) {
            console.error(`[i18n] Failed to change language to ${lang}:`, error);
            return false;
        }
    }

    /**
     * Get the current language
     * @returns {string} - Current language code
     */
    function getCurrentLanguage() {
        return currentLanguage;
    }

    /**
     * Get list of supported languages
     * @returns {Array<Object>} - Array of language objects with code and name
     */
    function getSupportedLanguages() {
        return [
            { code: 'en', name: 'English' },
            { code: 'zh-CN', name: '简体中文' }
        ];
    }

    /**
     * Create a language selector element
     * @returns {HTMLElement} - Select element for language switching
     */
    function createLanguageSelector() {
        const select = document.createElement('select');
        select.id = 'language-selector';
        select.className = 'styled-select language-select';
        
        getSupportedLanguages().forEach(lang => {
            const option = document.createElement('option');
            option.value = lang.code;
            option.textContent = lang.name;
            if (lang.code === currentLanguage) {
                option.selected = true;
            }
            select.appendChild(option);
        });

        select.addEventListener('change', async (e) => {
            const success = await setLanguage(e.target.value);
            if (!success) {
                e.target.value = currentLanguage;
            }
        });

        return select;
    }

    // Public API
    return {
        init,
        t,
        setLanguage,
        getCurrentLanguage,
        getSupportedLanguages,
        createLanguageSelector,
        applyTranslations
    };
})();

// Auto-initialize when DOM is ready (if not using as module)
if (typeof document !== 'undefined') {
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', () => I18n.init());
    } else {
        // DOM already loaded
        I18n.init();
    }
}
