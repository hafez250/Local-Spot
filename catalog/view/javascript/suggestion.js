//  Suggestion Module JavaScript for OpenCart 4
(function($) {
    'use strict';

    var Suggestion = {
        init: function() {
            this.bindEvents();
            console.log(' Suggestion module initialized');
        },

        bindEvents: function() {
            $('.suggestion-module').on('click', 'button', this.handleClick.bind(this));
            $('.suggestion-module').on('change', 'input, select', this.handleChange.bind(this));
        },

        handleClick: function(e) {
            e.preventDefault();
            var $target = $(e.currentTarget);
            console.log(' Suggestion button clicked:', $target);
        },

        handleChange: function(e) {
            var $target = $(e.currentTarget);
            var value = $target.val();
            console.log(' Suggestion input changed:', value);
        },

        ajax: function(url, data, callback) {
            $.ajax({
                url: url,
                type: 'POST',
                data: data,
                dataType: 'json',
                beforeSend: function() {
                    console.log('AJAX request started');
                },
                success: function(response) {
                    if (callback && typeof callback === 'function') {
                        callback(response);
                    }
                },
                error: function(xhr, status, error) {
                    console.error(' Suggestion AJAX error:', error);
                }
            });
        }
    };

    // Initialize when document is ready
    $(document).ready(function() {
        Suggestion.init();
    });

    // Make available globally
    window.Suggestion = Suggestion;

})(jQuery);