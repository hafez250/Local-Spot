<?php
namespace Opencart\Catalog\Controller\Product;

class All extends \Opencart\System\Engine\Controller {

    public function index(): void {
        $this->load->language('product/category'); // use category language for buttons etc.
        $this->load->model('catalog/product');
        $this->load->model('tool/image');

        $wishlist_product_ids = [];

        if ($this->customer->isLogged()) {
            $this->load->model('account/wishlist');

            foreach ($this->model_account_wishlist->getWishlist($this->customer->getId()) as $wishlist_product) {
                $wishlist_product_ids[] = (int)$wishlist_product['product_id'];
            }
        } elseif (!empty($this->session->data['wishlist']) && is_array($this->session->data['wishlist'])) {
            $wishlist_product_ids = array_map('intval', $this->session->data['wishlist']);
        }

        $data['products'] = [];

        // Get all products (you can increase limit if needed)
        $filter_data = [
            'start' => 0,
            'limit' => 1000
        ];

        $results = $this->model_catalog_product->getProducts($filter_data);

        foreach ($results as $result) {

            // Prepare product data for your Twig card
            $price = $result['price']; // format as needed
            $special = $result['special']; // format as needed
            $thumb = $result['image'] ? $this->model_tool_image->resize($result['image'], 300, 300) : $this->model_tool_image->resize('placeholder.png', 300, 300);
            $product_id = (int)$result['product_id'];

            $data['products'][] = [
                'product_id' => $product_id,
                'thumb'      => $thumb,
                'name'       => $result['name'],
                'description'=> mb_substr(strip_tags(html_entity_decode($result['description'], ENT_QUOTES, 'UTF-8')), 0, 100) . '..',
                'price'      => $price,
                'special'    => $special,
                'href'       => $this->url->link('product/product', 'language=' . $this->config->get('config_language') . '&product_id=' . $product_id),
                'cart_add'   => $this->url->link('checkout/cart.add', 'language=' . $this->config->get('config_language')),
                'wishlist_add' => $this->url->link('account/wishlist.add', 'language=' . $this->config->get('config_language')),
                'wishlist_remove' => $this->url->link('account/wishlist.remove', 'language=' . $this->config->get('config_language')),
                'in_wishlist' => in_array($product_id, $wishlist_product_ids, true),
                'minimum'    => $result['minimum'] ?? 1,
                'review_status' => true, // set if reviews are enabled
                'rating'     => $result['rating'] ?? 0,
                'button_wishlist' => $this->language->get('button_wishlist')
            ];
        }

        $data['header'] = $this->load->controller('common/header');
        $data['footer'] = $this->load->controller('common/footer');

        $this->response->setOutput($this->load->view('product/all', $data));
    }
}
