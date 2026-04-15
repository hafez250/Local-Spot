<?php
namespace Opencart\Catalog\Controller\Extension\Suggestion\Module;

class Suggestion extends \Opencart\System\Engine\Controller {
    public function index(): string {
        // Check if module is enabled
        if (!$this->config->get('module_suggestion_status')) {
            return '';
        }

        // Load language file
        $this->load->language('extension/suggestion/module/suggestion');

        // ✅ Load the model from catalog/model/module/suggestion.php
        $this->load->model('module/suggestion');

        // Get customer ID if logged in
        $customer_id = 0;
        if ($this->customer->isLogged()) {
            $customer_id = (int)$this->customer->getId();
        }
        

        $current_product_id = 0;

if (isset($this->request->get['product_id'])) {
    $current_product_id = (int)$this->request->get['product_id'];
}



        // ✅ Call the model method correctly
      $products = $this->model_module_suggestion->getSuggestedProducts($customer_id);

// Remove current product from the recommended list
if ($current_product_id) {
    $products = array_filter($products, function($product) use ($current_product_id) {
        return $product['product_id'] != $current_product_id;
    });
}

// Limit to 4 products
$products = array_slice($products, 0, 4);

$data['products'] = [];

foreach ($products as $product) {
    $product_data = [
        'product_id' => $product['product_id'],
        'thumb'      => $product['thumb'],
        'name'       => $product['name'],
        'price'      => $product['price'],
        'special'    => $product['special'] ?? false,
        'href'       => $product['href']
    ];

    $data['products'][] = $this->load->controller('product/thumb', $product_data);
    
}


        // Language texts
        $data['heading_title'] = $this->language->get('heading_title');
        $data['text_empty'] = $this->language->get('text_empty');

        // ✅ Render the view from your extension folder
        return $this->load->view('extension/suggestion/module/suggestion', $data);
    }
}
