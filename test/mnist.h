/*
 * mnist.h
 */
#ifndef __MNIST_H__
#define __MNIST_H__

class mnist_label {
public:
    mnist_label();
    ~mnist_label();

    void read_file(const char* name);
    unsigned char* get_label() const;
    unsigned int get_size() const;
private:
    unsigned char* label_vector;
    unsigned int label_number;
};

class mnist_image {
public:
    typedef unsigned char* pixel;
    mnist_image();
    ~mnist_image();

    void read_file(const char* name);
    pixel* get_image() const;
    unsigned int get_size() const;
    unsigned int get_row_size() const;
    unsigned int get_column_size() const;
private:
    pixel* image_vector;
    unsigned int row_size;
    unsigned int column_size;
    unsigned int image_number;
};

#endif /* __MNIST_H__ */
