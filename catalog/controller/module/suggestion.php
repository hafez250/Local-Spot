<?php
namespace Opencart\Catalog\Controller\Extension\Suggestion\Module;

class Suggestion extends \Opencart\System\Engine\Controller {

    public function index(): string {
        if (!$this->config->get('module_suggestion_status')) {
            return '';
        }

        $this->load->language('extension/suggestion/module/suggestion');

        $this->document->addStyle('catalog/view/stylesheet/suggestion.css');
        $this->document->addScript('catalog/view/javascript/suggestion.js');
        $data['text_content'] = $this->language->get('text_content');
        $data['heading_title'] = $this->language->get('heading_title');
    

        return $this->load->view('extension/suggestion/module/suggestion', $data);
    }
}
