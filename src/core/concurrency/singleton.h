//
// Created by Zero on 29/04/2022.
//

#pragma once


namespace ocarina {
template<class T>
class Singleton : public concepts::Noncopyable {
public:
    static T* instance()
    {
        static T _instance;
        return &_instance;
    }

protected:
    Singleton() = default;
};
}// namespace ocarina
