// Similar Module JavaScript for OpenCart 4
(function($) {
    'use strict';

    var Similar = {
        init: function() {
            this.bindEvents();
            console.log('Similar module initialized');
        },

        bindEvents: function() {
            $('.similar-module').on('click', 'button', this.handleClick.bind(this));
            $('.similar-module').on('change', 'input, select', this.handleChange.bind(this));
        },

        handleClick: function(e) {
            e.preventDefault();
            var $target = $(e.currentTarget);
            console.log('Similar button clicked:', $target);
        },

        handleChange: function(e) {
            var $target = $(e.currentTarget);
            var value = $target.val();
            console.log('Similar input changed:', value);
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
                    console.error('Similar AJAX error:', error);
                }
            });
        }
    };

    // Initialize when document is ready
    $(document).ready(function() {
        Similar.init();
    });

    // Make available globally
    window.Similar = Similar;

})(jQuery);