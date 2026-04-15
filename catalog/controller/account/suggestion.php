<?php
namespace Opencart\Catalog\Controller\Account;

class Suggestion extends \Opencart\System\Engine\Controller {
    public function index(): void {
        // Must be logged in
        if (!$this->customer->isLogged()) {
            $this->response->redirect($this->url->link('account/login', '', true));
        }

        $this->load->language('account/suggestion');
        $this->document->setTitle($this->language->get('heading_title'));

        // Load model
        $this->load->model('module/suggestion');

        // Get suggested products
        $products = $this->model_module_suggestion->getSuggestedProducts($this->customer->getId());

        // Render using the same product card as category/home
        $data['products'] = [];
        foreach ($products as $product) {
            $product_data = [
                'product_id'  => $product['product_id'] ?? 0,
                'name'        => $product['name'] ?? '',
                'description' => '',
                'thumb'       => $product['thumb'] ?? '',
                'price'       => $product['price'] ?? false,
                'special'     => false,
                'tax'         => false,
                'minimum'     => $product['minimum'] ?? 1,
                'href'        => $product['href'] ?? ''
            ];

            $data['products'][] = $this->load->controller('product/thumb', $product_data);
        }



        // Common components
        $data['header'] = $this->load->controller('common/header');
        $data['footer'] = $this->load->controller('common/footer');
        $data['column_left'] = $this->load->controller('common/column_left');

        // Language texts
        $data['heading_title'] = $this->language->get('heading_title');
        $data['text_empty'] = $this->language->get('text_empty');

        // Render view
        $this->response->setOutput($this->load->view('account/suggestion', $data));
    }
}
