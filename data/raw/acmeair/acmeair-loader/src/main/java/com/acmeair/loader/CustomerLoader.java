package com.acmeair.loader;

import com.acmeair.entities.Customer;
import com.acmeair.entities.CustomerAddress;
import com.acmeair.entities.Customer.PhoneType;
import com.acmeair.service.CustomerService;

public class CustomerLoader {

    private CustomerService customerService; // 默认 null

    public CustomerLoader() {
        // 默认构造函数
        this.customerService = null;
    }

    // 注入 CustomerService
    public void setCustomerService(CustomerService service) {
        this.customerService = service;
    }

    public void loadCustomers(long numCustomers) {
        if (customerService == null) {
            throw new IllegalStateException("CustomerService is not set!");
        }

        CustomerAddress address = customerService.createAddress("123 Main St.", null, "Anytown", "NC", "USA", "27617");
        for (long ii = 0; ii < numCustomers; ii++) {
            customerService.createCustomer(
                    "uid" + ii + "@email.com",
                    "password",
                    Customer.MemberShipStatus.GOLD,
                    1000000,
                    1000,
                    "919-123-4567",
                    PhoneType.BUSINESS,
                    address
            );
        }
    }
}
