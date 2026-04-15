<?php
namespace Extension\Opencart\Catalog\Controller\Module;

class Recommendation extends \Opencart\System\Engine\Controller {
    public function index(): string {
        // Load language
$this->load->language('module/recommendation'); // language file in your extension folder

        // Load models
        $this->load->model('catalog/product');
        $this->load->model('tool/image');
        $this->load->model('account/order');

        $data['products'] = [];
        $limit = 4; // Number of recommended products to show
        $recommended_category_id = 0;

        // Only for logged-in customers
        if ($this->customer->isLogged()) {
            $orders = $this->model_account_order->getOrders();
            $category_count = [];

            foreach ($orders as $order) {
                $products = $this->model_account_order->getOrderProducts($order['order_id']);

                foreach ($products as $product) {
                    $categories = $this->model_catalog_product->getCategories($product['product_id']);
                    foreach ($categories as $cat) {
                        if (!isset($category_count[$cat['category_id']])) {
                            $category_count[$cat['category_id']] = 0;
                        }
                        $category_count[$cat['category_id']]++;
                    }
                }
            }

            // Pick top category
            if ($category_count) {
                arsort($category_count);
                $recommended_category_id = array_key_first($category_count);
            }
        }

        // Get products from top category
        if ($recommended_category_id) {
            $filter_data = [
                'filter_category_id' => $recommended_category_id,
                'start'              => 0,
                'limit'              => $limit
            ];

            $results = $this->model_catalog_product->getProducts($filter_data);

            foreach ($results as $result) {
                $image = (is_file(DIR_IMAGE . $result['image']))
                    ? $this->model_tool_image->resize($result['image'], 200, 200)
                    : $this->model_tool_image->resize('placeholder.png', 200, 200);

                $data['products'][] = [
                    'product_id' => $result['product_id'],
                    'thumb'      => $image,
                    'name'       => $result['name'],
                    'price'      => $this->currency->format($result['price'], $this->session->data['currency']),
                    'href'       => $this->url->link('product/product', 'product_id=' . $result['product_id'])
                ];
            }
        }

        // Load Twig template
return $this->load->view('extension/module/recommendation', $data); // twig in your extension folder
    }
}
