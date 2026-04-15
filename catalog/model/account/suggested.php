<?php
namespace Opencart\Catalog\Controller\Account;

class Suggestion extends \Opencart\System\Engine\Controller {
    public function index(): void {
        // Load language file
        $this->load->language('account/suggestion');

        // Load model
        $this->load->model('module/suggestion');

        // Get suggested products
        $data['products'] = $this->model_module_suggestion->getSuggestedProducts();

        // Set up page
        $data['heading_title'] = $this->language->get('heading_title');
        $data['text_empty'] = $this->language->get('text_empty');

        // Common layout parts
        $data['column_left']   = $this->load->controller('common/column_left');
        $data['column_right']  = $this->load->controller('common/column_right');
        $data['content_top']   = $this->load->controller('common/content_top');
        $data['content_bottom']= $this->load->controller('common/content_bottom');
        $data['footer']        = $this->load->controller('common/footer');
        $data['header']        = $this->load->controller('common/header');

        // Output view
        $this->response->setOutput($this->load->view('account/suggestion', $data));
    }
}
